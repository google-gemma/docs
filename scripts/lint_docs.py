#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Documentation linter for the Gemma docs port.

Guards the invariants in PLAN.md so cleanup work can't silently regress. Every check is a
small, independent function registered in CHECKS; add a new one by writing the function and
registering it, nothing else.

    uv run python scripts/lint_docs.py                    # all checks, full report
    uv run python scripts/lint_docs.py --check links      # one check
    uv run python scripts/lint_docs.py --check fences,admonitions
    uv run python scripts/lint_docs.py --list             # show available checks
    uv run python scripts/lint_docs.py --format json      # machine-readable

Baseline
--------
The port starts with a large backlog of known violations. A baseline file records them so CI
fails only on *new* violations while the backlog is burned down:

    uv run python scripts/lint_docs.py --write-baseline   # snapshot current violations
    uv run python scripts/lint_docs.py --baseline .lintbaseline.json

Findings in the baseline are reported as "known" and do not affect the exit code. Findings that
have been fixed since the baseline was written are reported too, as a nudge to shrink it.
Shrinking the baseline to zero is the definition of done for Phase 0.

Exit codes: 0 = no new findings, 1 = new findings, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontmatter_schema  # noqa: E402  (needs the path above)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
CONFIG_FILE = REPO_ROOT / "zensical.toml"
DEFAULT_BASELINE = REPO_ROOT / ".lintbaseline.json"

# --- Thresholds -------------------------------------------------------------------------------

# Body text below this is a stub, not a page.
MIN_BODY_CHARS = 400

# Above this, something went wrong in conversion (usually inlined base64 images).
MAX_FILE_BYTES = 60_000

# The frontmatter schema lives in scripts/frontmatter_schema.py so the linter and the
# backfill script can never disagree about it. See PLAN.md, "Frontmatter schema".

# The site being replaced. Any link here is a leak.
UPSTREAM_DOCS = "ai.google.dev/gemma/docs"

# Image hosts allowed to stay remote FOR NOW. The visual guides are imported with their
# Substack CDN URLs so the pages can be reviewed before anyone spends time on artwork.
# Vendoring them is PLAN.md Step 6.1 -- empty this tuple when that lands.
PENDING_IMAGE_HOSTS = ("substackcdn.com", "substack-post-media")

# Devsite callouts that should have become admonitions during conversion.
DEVSITE_CALLOUTS = ("Note:", "Key Point:", "Caution:", "Warning:", "Important:", "Tip:")

# GitHub-style callouts, which devsite also emits: `> [!NOTE]`, sometimes double-quoted as
# `> > [!NOTE]`. These must become `!!! note` too.
GITHUB_CALLOUT = re.compile(r"^\s*(?:>\s*)+\[!(NOTE|IMPORTANT|WARNING|TIP|CAUTION)\]", re.IGNORECASE)

# Signals that a code block really is Python. Absence of all of these in a block tagged
# `python` means prose or a prompt template got captured by the converter.
PY_SIGNALS = re.compile(
    r"\b(import|from|def|class|return|print|lambda|None|True|False|self|async|await|yield)\b"
    r"|[=<>]=?\s"
    r"|\w\("
    r"|\)\s*[:.]"
    r"|^\s*[#@]"
    r"|\[[\"'\d]",
    re.MULTILINE,
)

# Two or more consecutive word-like tokens -- i.e. it reads as a sentence.
PROSE_SIGNAL = re.compile(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}")


# --- Model ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One violation, anchored to a file and (where meaningful) a line."""

    check: str
    path: str  # repo-relative, forward slashes
    line: int  # 1-indexed; 0 when the finding is about the file as a whole
    message: str

    def key(self) -> str:
        """Stable identity for baseline matching.

        Deliberately excludes the line number so that unrelated edits above a known finding
        don't resurrect it as "new".
        """
        return f"{self.check}|{self.path}|{self.message}"

    def location(self) -> str:
        return f"{self.path}:{self.line}" if self.line else self.path


@dataclass
class Page:
    """A markdown page with its frontmatter split out."""

    path: Path
    rel: str  # relative to docs/, forward slashes
    text: str
    frontmatter: dict = field(default_factory=dict)
    has_frontmatter: bool = False
    body: str = ""
    body_offset: int = 0  # lines consumed by the frontmatter block

    @property
    def repo_rel(self) -> str:
        return f"docs/{self.rel}"

    def body_lines(self) -> Iterator[tuple[int, str]]:
        """Yield (1-indexed file line number, line) for the body only."""
        for i, line in enumerate(self.body.splitlines(), start=self.body_offset + 1):
            yield i, line


@dataclass
class Context:
    """Everything the checks need, loaded once."""

    pages: list[Page]
    nav_entries: list[str]  # docs-relative paths referenced by zensical.toml nav
    config: dict

    @property
    def page_paths(self) -> set[str]:
        return {p.rel for p in self.pages}


# --- Loading ----------------------------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict, bool, str, int]:
    """Split a YAML frontmatter block off the top of a document.

    Returns (mapping, had_frontmatter, body, lines_consumed). Parsed as real YAML: the schema
    validates list-valued keys (`models`, `frameworks`), which a line-splitting parser reads
    as the string "[gemma-4]".
    """
    if not text.startswith("---"):
        return {}, False, text, 0

    lines = text.splitlines()
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            block = "\n".join(lines[1:idx])
            try:
                mapping = yaml.safe_load(block) or {}
            except yaml.YAMLError as exc:
                mapping = {"__error__": str(exc)}
            if not isinstance(mapping, dict):
                mapping = {"__error__": "frontmatter is not a mapping"}
            body = "\n".join(lines[idx + 1 :])
            return mapping, True, body, idx + 1

    # Unterminated block: treat the whole file as body.
    return {}, False, text, 0


def load_pages() -> list[Page]:
    pages = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        fm, has_fm, body, offset = parse_frontmatter(text)
        pages.append(
            Page(
                path=path,
                rel=path.relative_to(DOCS_DIR).as_posix(),
                text=text,
                frontmatter=fm,
                has_frontmatter=has_fm,
                body=body,
                body_offset=offset,
            )
        )
    return pages


def walk_nav(node) -> Iterator[str]:
    """Yield every markdown path in zensical.toml's nested nav structure."""
    if isinstance(node, str):
        if node.endswith(".md"):
            yield node
    elif isinstance(node, list):
        for item in node:
            yield from walk_nav(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from walk_nav(value)


def load_context() -> Context:
    config = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    nav = config.get("project", {}).get("nav", [])
    return Context(pages=load_pages(), nav_entries=list(walk_nav(nav)), config=config)


FENCE_OPEN = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


def iter_code_blocks(page: Page) -> Iterator[tuple[int, str, str]]:
    """Yield (start line, info string, block content) for each fenced code block."""
    open_at: int | None = None
    active_fence = ""
    info = ""
    buf: list[str] = []
    for lineno, line in page.body_lines():
        m = FENCE_OPEN.match(line)
        if open_at is None:
            if m:
                open_at, active_fence, info, buf = lineno, m.group(2), m.group(3).strip(), []
            continue
        if m and m.group(2) == active_fence and not m.group(3).strip():
            yield open_at, info, "\n".join(buf)
            open_at = None
        else:
            buf.append(line)


INLINE_CODE = re.compile(r"`+[^`]*`+")


def strip_inline_code(line: str) -> str:
    """Blank out `code spans`, keeping the line length so columns still line up.

    An inline span is code for the same reason a fenced block is: documentation *about* markup
    quotes it. Without this, `[Video](youtube…)` in the components guide reads as a broken link.
    """
    return INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line)


def outside_code(page: Page) -> Iterator[tuple[int, str]]:
    """Yield body lines that are not inside a fenced code block, with code spans blanked."""
    active_fence: str | None = None
    for lineno, line in page.body_lines():
        m = FENCE_OPEN.match(line)
        if active_fence is None:
            if m:
                active_fence = m.group(2)
                continue
            yield lineno, strip_inline_code(line)
        elif m and m.group(2) == active_fence and not m.group(3).strip():
            active_fence = None
            continue


# --- Checks -----------------------------------------------------------------------------------

MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
HTML_SRC = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
EXPLICIT_ID = re.compile(r"(?:\{[^}]*#([\w-]+)[^}]*\}|\bid=[\"']([^\"']+)[\"'])")


def slugify_heading(heading: str) -> str:
    """Mirror Python-Markdown's default TOC slug generation."""
    value = re.sub(r"<[^>]+>", "", heading)
    value = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[*`_]", "", value)
    value = re.sub(r"\{:?[^}]*\}", "", value)
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def anchors_for(page: Page) -> set[str]:
    """Return heading slugs and explicit IDs available on a page."""
    anchors: set[str] = set()
    for _, line in outside_code(page):
        heading = HEADING.match(line)
        if heading:
            anchors.add(slugify_heading(heading.group(2)))
        for explicit in EXPLICIT_ID.finditer(line):
            anchors.add(explicit.group(1) or explicit.group(2))
    return anchors


def check_links(ctx: Context) -> Iterable[Finding]:
    """No prose links back to the documentation site we're replacing.

    Code blocks are exempt, and deliberately so. Every upstream URL inside a snippet here is
    either sample data the reader is meant to fetch (`img_url = "https://.../cat.png"`) or a
    BibTeX citation, which must keep pointing at the canonical public URL. Rewriting those to
    local paths would break copy-paste and misattribute citations.
    """
    for page in ctx.pages:
        for lineno, line in outside_code(page):
            if UPSTREAM_DOCS in line:
                for match in re.finditer(rf"https?://{re.escape(UPSTREAM_DOCS)}[^\s)\"'>]*", line):
                    yield Finding(
                        "links", page.repo_rel, lineno,
                        f"link to upstream docs: {match.group(0)}",
                    )


# `[text][id]` -- a reference link. An id that resolves to nothing renders as literal text
# rather than failing, so nobody notices until a reader sees the brackets on the page.
REFERENCE_USE = re.compile(r"\[[^\]]+\]\[([a-z0-9][a-z0-9-]*)\]")
REFERENCE_DEF = re.compile(r"^\[([^\]]+)\]:\s+\S", re.MULTILINE)


def check_references(ctx: Context) -> Iterable[Finding]:
    """Every `[text][id]` must resolve, from data/links.yml or from the page itself."""
    import generate_snippets

    shared = set(generate_snippets.link_data())
    for page in ctx.pages:
        local = set(REFERENCE_DEF.findall(page.path.read_text(encoding="utf-8")))
        for lineno, line in outside_code(page):
            for match in REFERENCE_USE.finditer(line):
                name = match.group(1)
                if name not in shared and name not in local:
                    yield Finding(
                        "references", page.repo_rel, lineno,
                        f"unknown reference '[{name}]' -- add it to data/links.yml or define it "
                        "on the page; an unresolved one renders as literal text",
                    )


def check_icons(ctx: Context) -> Iterable[Finding]:
    """Every icon a page asks for must be vendored under overrides/.icons/.

    `icon()` falls back to the installed Zensical package, so an un-vendored icon still renders
    while authoring -- and then fails in CI, where only PyYAML and Jinja2 are installed. This
    check is what turns that into an obvious message instead of a stack trace.
    """
    import vendor_icons

    for name in sorted(vendor_icons.missing()):
        yield Finding(
            "icons", "overrides/.icons", 0,
            f"'{name}' is referenced but not vendored -- run scripts/vendor_icons.py",
        )


def check_images(ctx: Context) -> Iterable[Finding]:
    """Images must be vendored into docs/assets/, not hotlinked.

    Scans outside code blocks only -- markdown image syntax inside a snippet is code, not
    an image reference (e.g. an f-string like `{render_inline(image, resize=(64,64))}`).
    """
    for page in ctx.pages:
        for lineno, line in outside_code(page):
            for pattern in (MD_IMAGE, HTML_SRC):
                for match in pattern.finditer(line):
                    src = match.group(1)
                    if src.startswith("data:"):
                        yield Finding(
                            "images", page.repo_rel, lineno,
                            "inline base64 image -- extract to docs/assets/",
                        )
                    elif src.startswith(("http://", "https://")):
                        if any(host in src for host in PENDING_IMAGE_HOSTS):
                            continue  # see PENDING_IMAGE_HOSTS
                        yield Finding(
                            "images", page.repo_rel, lineno,
                            f"hotlinked image: {src}",
                        )
                    else:
                        target = (page.path.parent / src.split("#")[0]).resolve()
                        if not target.exists():
                            yield Finding(
                                "images", page.repo_rel, lineno,
                                f"missing local image: {src}",
                            )


def check_internal(ctx: Context) -> Iterable[Finding]:
    """Relative links must resolve to a real file and fragment."""
    pages_by_path = {page.path.resolve(): page for page in ctx.pages}
    anchors_by_path = {path: anchors_for(page) for path, page in pages_by_path.items()}

    for page in ctx.pages:
        for lineno, line in outside_code(page):
            for match in MD_LINK.finditer(line):
                target = match.group(1)
                if target.startswith(("http://", "https://", "mailto:", "data:", "/")):
                    continue
                if line[match.start()] == "!":  # images handled by check_images
                    continue

                path_part, separator, fragment = target.partition("#")
                resolved = page.path.resolve() if not path_part else (page.path.parent / path_part).resolve()
                if not resolved.exists():
                    yield Finding(
                        "internal", page.repo_rel, lineno,
                        f"broken internal link: {target}",
                    )
                    continue

                if separator and fragment and resolved in anchors_by_path:
                    if fragment not in anchors_by_path[resolved]:
                        yield Finding(
                            "internal", page.repo_rel, lineno,
                            f"broken internal anchor: {target}",
                        )


def check_stubs(ctx: Context) -> Iterable[Finding]:
    """Every page in the nav must have real content."""
    for page in ctx.pages:
        # A page rendered by a template holds its content there, not in the body.
        if page.frontmatter.get("template"):
            continue
        if "Content coming soon" in page.body:
            yield Finding("stubs", page.repo_rel, 0, "placeholder text: 'Content coming soon'")
            continue
        if "example.com" in page.body:
            yield Finding("stubs", page.repo_rel, 0, "placeholder link: example.com")
        # Strip headings and blank lines before measuring.
        prose = "\n".join(
            line for line in page.body.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if len(prose) < MIN_BODY_CHARS:
            yield Finding(
                "stubs", page.repo_rel, 0,
                f"stub: {len(prose)} chars of content (minimum {MIN_BODY_CHARS})",
            )


def first_heading(page: Page) -> str | None:
    """The page's own H1, or None. Fenced blocks are skipped -- a shell comment such as
    `# Install PyTorch` is not a heading, and reading it as one mistitles the page."""
    in_fence = False
    for line in page.body.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence or not line.startswith("# "):
            continue
        text = re.sub(r"\s*\{[^}]*\}\s*$", "", line[2:])
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[*`_]", "", text).strip()
        if text:
            return text
    return None


def check_frontmatter(ctx: Context) -> Iterable[Finding]:
    """Every page carries agent-readable frontmatter (see PLAN.md, "Frontmatter schema")."""
    for page in ctx.pages:
        if not page.has_frontmatter:
            yield Finding("frontmatter", page.repo_rel, 0, "no frontmatter block")
            continue
        if "__error__" in page.frontmatter:
            yield Finding(
                "frontmatter", page.repo_rel, 0, f"unparseable: {page.frontmatter['__error__']}"
            )
            continue
        for problem in frontmatter_schema.validate(page.frontmatter, page_paths=ctx.page_paths):
            yield Finding("frontmatter", page.repo_rel, 0, problem.message)

        # A page with no H1 of its own gets one synthesised from `title` (Zensical emits
        # `<h1 id="__skip">`). On those pages the title is visible content, not just metadata,
        # so it has to say what the page already said. Where the page *has* an H1, the title
        # only reaches the browser tab and llms.txt and may be more specific than the heading.
        if first_heading(page) is None and "title" not in page.frontmatter:
            yield Finding(
                "frontmatter",
                page.repo_rel,
                0,
                "page has no H1, so `title` is rendered as the heading -- it must be set",
            )


HIDDEN_PREFIXES = ("contributing/",)


def check_orphans(ctx: Context) -> Iterable[Finding]:
    """Nav and filesystem must agree."""
    nav = set(ctx.nav_entries)
    visible_pages = {rel for rel in ctx.page_paths if not rel.startswith(HIDDEN_PREFIXES)}
    for rel in sorted(visible_pages - nav):
        yield Finding("orphans", f"docs/{rel}", 0, "page exists but is not in nav")
    for rel in sorted(nav - ctx.page_paths):
        yield Finding("orphans", "zensical.toml", 0, f"nav entry has no file: {rel}")
    for rel in sorted(nav & (ctx.page_paths - visible_pages)):
        yield Finding("orphans", "zensical.toml", 0, f"hidden page must not be in nav: {rel}")
    seen: set[str] = set()
    for rel in ctx.nav_entries:
        if rel in seen:
            yield Finding("orphans", "zensical.toml", 0, f"page listed twice in nav: {rel}")
        seen.add(rel)


def check_fences(ctx: Context) -> Iterable[Finding]:
    """Catch converter damage: prose captured as Python, and code blocks split in half."""
    for page in ctx.pages:
        if page.body.count("```") % 2:
            yield Finding("fences", page.repo_rel, 0, "unbalanced code fences")

        # Two adjacent *output* blocks separated by nothing but blank lines are one stream of
        # terminal output that got split -- always safe to merge, and convert_notebook does.
        #
        # Deliberately not applied to code blocks: two adjacent `python` blocks are usually
        # consecutive notebook cells, where the first happens to end with captured output.
        # Merging those would be wrong. The blank-lines-inside-one-cell defect that motivated
        # this check is fixed in the converter and pinned by a golden test.
        blocks = list(iter_code_blocks(page))
        body = page.body.splitlines()
        for (start_a, info_a, code_a), (start_b, info_b, _) in zip(blocks, blocks[1:]):
            if info_a != info_b or ".output" not in info_a:
                continue
            end_a = start_a + len(code_a.splitlines()) + 1  # closing fence
            between = body[end_a - page.body_offset : start_b - page.body_offset - 1]
            if all(not line.strip() for line in between):
                yield Finding(
                    "fences", page.repo_rel, start_b,
                    f"`{info_a}` block split in two -- likely blank lines inside one block",
                )

        for lineno, info, code in blocks:
            lang = info.split()[0].lower() if info else ""
            if lang not in ("python", "py"):
                continue
            stripped = code.strip()
            if not stripped or PY_SIGNALS.search(stripped) or not PROSE_SIGNAL.search(stripped):
                continue
            preview = " ".join(stripped.split())[:60]
            kind = "prompt template" if "{" in stripped else "prose"
            yield Finding(
                "fences", page.repo_rel, lineno,
                f"{kind} tagged as python: '{preview}...'",
            )


def check_model_swap(ctx: Context) -> Iterable[Finding]:
    """A swappable engine must not also appear in unacknowledged static code blocks."""
    data = yaml.safe_load((REPO_ROOT / "data" / "models.yml").read_text(encoding="utf-8"))
    engines: dict[str, set[str]] = {}
    index: list[tuple[str, str, str]] = []
    for size in data.get("gemma4", {}).get("sizes", []):
        size_id = str(size["id"])
        for engine, identifier in size.get("ids", {}).items():
            engines.setdefault(str(engine), set()).add(str(identifier))
            index.append((str(identifier), str(engine), size_id))
    index.sort(key=lambda item: len(item[0]), reverse=True)
    pattern = re.compile("|".join(re.escape(text) for text, _, _ in index))
    by_text = {text: (engine, size_id) for text, engine, size_id in index}
    unrewritten_re = re.compile(
        r"\bgemma[-_]?4[-_:(](?:e2b|e4b|12b|26b(?:[-_]a4b)?|31b)[\w.-]*",
        re.IGNORECASE,
    )

    for page in ctx.pages:
        blocks = list(iter_code_blocks(page))
        marked = [(line, info, code) for line, info, code in blocks if ".model-swap" in info]
        if not marked:
            continue

        active: set[str] = set()
        page_default_size: str | None = None
        for line, _, code in marked:
            matches = [m.group(0) for m in pattern.finditer(code)]
            if not matches:
                yield Finding(
                    "model-swap",
                    page.repo_rel,
                    line,
                    ".model-swap block contains no swappable model identifier from data/models.yml",
                )
                continue
            for match in matches:
                engine, size_id = by_text[match]
                active.add(engine)
                if page_default_size is None:
                    page_default_size = size_id
                elif size_id != page_default_size:
                    yield Finding(
                        "model-swap",
                        page.repo_rel,
                        line,
                        f".model-swap identifier '{match}' targets '{size_id}' "
                        f"but page default is '{page_default_size}'",
                    )
            remainder = pattern.sub("", code)
            for leftover in unrewritten_re.finditer(remainder):
                yield Finding(
                    "model-swap",
                    page.repo_rel,
                    line,
                    f"unrewritten model identifier '{leftover.group(0)}' inside .model-swap block",
                )

        for line, info, code in blocks:
            if ".model-swap" in info or ".model-fixed" in info or ".output" in info:
                continue
            for engine in active:
                identifiers = engines[engine]
                hits = [identifier for identifier in identifiers if identifier in code]
                if not hits and any(":" in identifier for identifier in identifiers):
                    prefixes = {identifier.split(":", 1)[0] for identifier in identifiers}
                    hits = [
                        prefix
                        for prefix in prefixes
                        if re.search(rf"(?<![\w/-]){re.escape(prefix)}(?=[:\s\"'])", code)
                    ]
                if hits:
                    yield Finding(
                        "model-swap",
                        page.repo_rel,
                        line,
                        f"{engine} identifier '{sorted(hits)[0]}' is outside a .model-swap "
                        "block -- mark it swappable or explicitly .model-fixed",
                    )


def check_admonitions(ctx: Context) -> Iterable[Finding]:
    """Devsite and GitHub callouts should both be Zensical admonitions."""
    for page in ctx.pages:
        for lineno, line in outside_code(page):
            github = GITHUB_CALLOUT.match(line)
            if github:
                yield Finding(
                    "admonitions", page.repo_rel, lineno,
                    f"unconverted callout: '[!{github.group(1).upper()}]' -- use !!! admonition",
                )
                continue
            stripped = line.lstrip("> ").lstrip()
            for callout in DEVSITE_CALLOUTS:
                if stripped.startswith(callout):
                    yield Finding(
                        "admonitions", page.repo_rel, lineno,
                        f"unconverted devsite callout: '{callout}' -- use !!! admonition",
                    )
                    break


def check_size(ctx: Context) -> Iterable[Finding]:
    """Oversized files mean conversion inlined something it shouldn't have."""
    for page in ctx.pages:
        size = page.path.stat().st_size
        if size > MAX_FILE_BYTES:
            yield Finding(
                "size", page.repo_rel, 0,
                f"{size:,} bytes exceeds {MAX_FILE_BYTES:,} -- check for inlined assets",
            )


def _figure(value) -> str | None:
    """Render a memory figure the way the ported table writes it: 11.4, not 11.400000000001."""
    if value is None:
        return None
    text = f"{float(value):g}"
    return text


def check_modeldata(ctx: Context) -> Iterable[Finding]:
    """Keep data/models.yml in step with the ported memory table it was copied from.

    models/gemma-4/index.md is a straight port, so its table can't be generated from the data
    file -- re-porting would overwrite the template. Instead the two are compared here, so
    upstream revising a figure shows up as a finding rather than a silent contradiction.
    """
    data_file = REPO_ROOT / "data" / "models.yml"
    page_path = DOCS_DIR / "models" / "gemma-4" / "index.md"
    if not data_file.exists() or not page_path.exists():
        return

    # Parsed as YAML, scoped to gemma4.sizes. An earlier line-scraper matched every `name:` in
    # the file, so adding the `families:` block made it demand a memory row for "DataGemma".
    data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    declared: list[tuple[str, dict[str, str]]] = [
        (
            str(size["name"]),
            {key: _figure(size.get("memory", {}).get(key)) for key in ("bf16", "sfp8", "q4_0")},
        )
        for size in data.get("gemma4", {}).get("sizes", [])
    ]

    # Table rows look like: | Gemma 4 E2B | 11.4 GB | 5.7 GB | 2.9 GB | ...
    rows = {}
    for line in page_path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 4 and cells[0].startswith("Gemma 4 "):
            rows[cells[0].removeprefix("Gemma 4 ")] = [
                c.removesuffix(" GB").strip() for c in cells[1:4]
            ]

    for name, figures in declared:
        row = rows.get(name)
        if row is None:
            yield Finding(
                "modeldata", "data/models.yml", 0,
                f"'{name}' has no row in models/gemma-4/index.md",
            )
            continue
        for key, expected in zip(("bf16", "sfp8", "q4_0"), row):
            actual = figures.get(key)
            if actual and actual != expected:
                yield Finding(
                    "modeldata", "data/models.yml", 0,
                    f"{name} {key}: data says {actual}, ported table says {expected}",
                )


STYLESHEET_DIR = DOCS_DIR / "stylesheets"
TOKEN_STYLESHEET = "gemma.css"
MAX_STYLESHEET_LINES = 250
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def check_css(ctx: Context) -> Iterable[Finding]:
    """Design-system rules from PLAN.md Phase 1.

    Raw colours live in exactly one file, every sheet is registered, and none grows past the
    point where an agent can no longer read it end-to-end before editing.
    """
    if not STYLESHEET_DIR.exists():
        return

    registered = {
        Path(entry.split("?", 1)[0]).name
        for entry in ctx.config.get("project", {}).get("extra_css", [])
    }

    for path in sorted(STYLESHEET_DIR.glob("*.css")):
        rel = f"docs/stylesheets/{path.name}"
        text = path.read_text(encoding="utf-8")

        if path.name not in registered:
            yield Finding("css", rel, 0, "stylesheet is not listed in extra_css")

        lines = text.splitlines()
        content_lines = re.sub(r"^\s*/\*[\s\S]*?Copyright[\s\S]*?\*/\s*", "", text).splitlines()
        if len(content_lines) > MAX_STYLESHEET_LINES:
            yield Finding(
                "css", rel, 0,
                f"{len(content_lines)} lines exceeds {MAX_STYLESHEET_LINES} -- split by concern",
            )

        if path.name == TOKEN_STYLESHEET:
            continue
        for number, line in enumerate(lines, start=1):
            if HEX_COLOUR.search(line):
                yield Finding(
                    "css", rel, number,
                    f"raw colour {HEX_COLOUR.search(line).group(0)} -- "
                    f"add a token to {TOKEN_STYLESHEET} and reference it",
                )

    for name in sorted(registered):
        if not (STYLESHEET_DIR / name).exists():
            yield Finding("css", "zensical.toml", 0, f"extra_css lists a missing file: {name}")

    # Template style blocks bypass every rule above. Presentation belongs in registered sheets
    # so its load order and token use remain testable.
    overrides_dir = REPO_ROOT / "overrides"
    if overrides_dir.exists():
        for path in sorted(overrides_dir.rglob("*.html")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if "<style" in line.lower():
                    yield Finding(
                        "css", path.relative_to(REPO_ROOT).as_posix(), number,
                        "inline style block -- move styles into a registered stylesheet",
                    )


def check_generated(ctx: Context) -> Iterable[Finding]:
    """Committed build output must match the data it was generated from.

    Zensical has no hook mechanism, so the page context header's model display names are
    generated from data/models.yml into a template partial. That partial is the one place in
    the repo where model data could silently go stale, hence this check.
    """
    import agent_docs
    import generate_model_names
    import generate_snippets
    import notebooks

    outputs = (
        (generate_model_names.TARGET, generate_model_names.render()),
        (generate_model_names.JS_TARGET, generate_model_names.render_js()),
        *agent_docs.generated_outputs(),
        *notebooks.generated_outputs(),
        *generate_snippets.generated_outputs(),
    )
    for target, expected in outputs:
        rel = target.relative_to(REPO_ROOT).as_posix()
        if not target.exists():
            yield Finding("generated", rel, 0, "missing -- rerun the corresponding generator")
        elif target.read_text(encoding="utf-8") != expected:
            yield Finding(
                "generated", rel, 0,
                "stale vs its canonical source -- rerun the corresponding generator",
            )


CHECKS: dict[str, Callable[[Context], Iterable[Finding]]] = {
    "links": check_links,
    "references": check_references,
    "icons": check_icons,
    "images": check_images,
    "internal": check_internal,
    "stubs": check_stubs,
    "frontmatter": check_frontmatter,
    "orphans": check_orphans,
    "fences": check_fences,
    "model-swap": check_model_swap,
    "admonitions": check_admonitions,
    "size": check_size,
    "modeldata": check_modeldata,
    "css": check_css,
    "generated": check_generated,
}

CHECK_HELP = {
    "links": "links back to ai.google.dev/gemma/docs",
    "references": "reference links that resolve to nothing",
    "icons": "icons referenced by a page but not vendored",
    "images": "hotlinked, inlined, or missing images",
    "internal": "relative links or fragments that don't resolve",
    "stubs": "placeholder pages and leftover example.com",
    "frontmatter": "missing or incomplete agent frontmatter",
    "orphans": "pages missing from nav, nav entries missing files",
    "fences": "prose or prompts captured as python code blocks",
    "model-swap": "model identifiers omitted from a page's opt-in switching",
    "admonitions": "unconverted devsite Note:/Caution: callouts",
    "size": "files large enough to indicate inlined assets",
    "modeldata": "data/models.yml vs the ported memory table",
    "css": "design-system rules: tokens, registration, file size",
    "generated": "committed build output that has drifted from its source data",
}


# --- Reporting --------------------------------------------------------------------------------


def load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("known", []))


def report_text(new: list[Finding], known: list[Finding], fixed: set[str], selected: list[str]) -> None:
    if new:
        print("NEW FINDINGS\n" + "=" * 78)
        by_check: dict[str, list[Finding]] = {}
        for f in new:
            by_check.setdefault(f.check, []).append(f)
        for check in selected:
            items = by_check.get(check)
            if not items:
                continue
            print(f"\n{check}  ({len(items)}) -- {CHECK_HELP[check]}")
            for f in items:
                print(f"  {f.location()}\n      {f.message}")
        print()

    print("SUMMARY\n" + "=" * 78)
    width = max(len(c) for c in selected)
    new_by_check: dict[str, int] = {}
    known_by_check: dict[str, int] = {}
    for f in new:
        new_by_check[f.check] = new_by_check.get(f.check, 0) + 1
    for f in known:
        known_by_check[f.check] = known_by_check.get(f.check, 0) + 1

    print(f"  {'check':<{width}}   new    known")
    for check in selected:
        n, k = new_by_check.get(check, 0), known_by_check.get(check, 0)
        flag = "" if not n else "  <--"
        print(f"  {check:<{width}}  {n:>4}  {k:>7}{flag}")
    print(f"  {'-' * (width + 16)}")
    print(f"  {'total':<{width}}  {len(new):>4}  {len(known):>7}")

    if fixed:
        print(f"\n  {len(fixed)} baseline finding(s) no longer present -- rerun with "
              f"--write-baseline to shrink the baseline.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint the Gemma docs against the invariants in PLAN.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check", default="all",
        help="comma-separated check names, or 'all' (default). See --list.",
    )
    parser.add_argument("--list", action="store_true", help="list available checks and exit")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--baseline", nargs="?", const=str(DEFAULT_BASELINE), default=None,
        help=f"treat findings in this file as known (default file: {DEFAULT_BASELINE.name})",
    )
    parser.add_argument(
        "--write-baseline", action="store_true",
        help="write current findings to the baseline file and exit 0",
    )
    args = parser.parse_args(argv)

    if args.list:
        width = max(len(c) for c in CHECKS)
        for name in CHECKS:
            print(f"  {name:<{width}}  {CHECK_HELP[name]}")
        return 0

    if args.check == "all":
        selected = list(CHECKS)
    else:
        selected = [c.strip() for c in args.check.split(",") if c.strip()]
        unknown = [c for c in selected if c not in CHECKS]
        if unknown:
            parser.error(f"unknown check(s): {', '.join(unknown)}. Use --list to see options.")

    if not CONFIG_FILE.exists():
        parser.error(f"config not found: {CONFIG_FILE}")

    ctx = load_context()
    findings = [f for name in selected for f in CHECKS[name](ctx)]
    findings.sort(key=lambda f: (f.check, f.path, f.line))

    if args.write_baseline:
        # A partial run would drop every unselected check's entries from the baseline.
        if set(selected) != set(CHECKS):
            parser.error("--write-baseline requires all checks; drop --check and rerun.")
        DEFAULT_BASELINE.write_text(
            json.dumps(
                {
                    "_comment": "Known violations at baseline time. Shrink to zero; never grow. "
                                "Regenerate with: uv run python scripts/lint_docs.py --write-baseline",
                    "checks": selected,
                    "count": len(findings),
                    "known": sorted({f.key() for f in findings}),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote baseline with {len(findings)} finding(s) to {DEFAULT_BASELINE.name}")
        return 0

    baseline = load_baseline(Path(args.baseline)) if args.baseline else set()
    new = [f for f in findings if f.key() not in baseline]
    known = [f for f in findings if f.key() in baseline]

    # Only baseline entries belonging to the checks we actually ran can be judged fixed --
    # otherwise `--check links` would report every unrelated baseline entry as resolved.
    scoped_baseline = {k for k in baseline if k.split("|", 1)[0] in selected}
    fixed = scoped_baseline - {f.key() for f in findings}

    if args.format == "json":
        print(json.dumps(
            {
                "new": [f.__dict__ for f in new],
                "known_count": len(known),
                "fixed_count": len(fixed),
            },
            indent=2,
        ))
    else:
        report_text(new, known, fixed, selected)

    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
