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

"""Build the Zensical site and add its agent-facing raw Markdown routes."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

import agent_docs

ANALYTICS_PROPERTY_RE = re.compile(
    r'(\[project\.extra\.analytics\][\s\S]*?\bproperty\s*=\s*)"[^"]*"'
)
SITE_URL_RE = re.compile(r'^(\s*site_url\s*=\s*)"[^"]*"', re.MULTILINE)
COLAB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
COLAB_BRANCH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
PUBLIC_COLAB_PREFIX = "https://colab.research.google.com/github/google-gemma/docs/blob/main/"
SITEMAP_URL_BLOCK_RE = re.compile(r"(<url>\s*<loc>([^<]+)</loc>)(\s*</url>)")


def inject_analytics_property(config_text: str, measurement_id: str) -> str:
    """Return config_text with project.extra.analytics.property set to measurement_id."""
    if not ANALYTICS_PROPERTY_RE.search(config_text):
        raise ValueError("Could not locate [project.extra.analytics] property in zensical.toml")
    return ANALYTICS_PROPERTY_RE.sub(rf'\g<1>"{measurement_id}"', config_text, count=1)


def inject_site_url(config_text: str, site_url: str) -> str:
    """Return config_text with project.site_url set to site_url."""
    if not SITE_URL_RE.search(config_text):
        raise ValueError("Could not locate site_url in zensical.toml")
    normalized = re.sub(r"(?<!:)//+", "/", site_url.strip())
    return SITE_URL_RE.sub(lambda m: f'{m.group(1)}"{normalized}"', config_text, count=1)


def rewrite_colab_links(site_dir: Path, repo: str, branch: str) -> int:
    """Rewrite canonical in-repo Colab links in built HTML files to target repo and branch."""
    clean_repo = repo.strip()
    clean_branch = branch.strip()
    if not COLAB_REPO_RE.fullmatch(clean_repo):
        raise ValueError(f"Invalid COLAB_REPO: {repo!r}")
    if (
        not COLAB_BRANCH_RE.fullmatch(clean_branch)
        or clean_branch.startswith("/")
        or clean_branch.endswith("/")
        or ".." in clean_branch
    ):
        raise ValueError(f"Invalid COLAB_BRANCH: {branch!r}")

    # URL-encode "/" as "%2F" so compound branch names stay a single path segment in Colab.
    encoded_branch = quote(clean_branch, safe="._-")
    target_prefix = f"https://colab.research.google.com/github/{clean_repo}/blob/{encoded_branch}/"
    if target_prefix == PUBLIC_COLAB_PREFIX:
        return 0

    rewritten_files = 0
    for html_path in site_dir.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        if PUBLIC_COLAB_PREFIX in text:
            html_path.write_text(
                text.replace(PUBLIC_COLAB_PREFIX, target_prefix),
                encoding="utf-8",
            )
            rewritten_files += 1
    return rewritten_files


def collect_git_lastmod_dates(repo_root: Path) -> dict[str, str]:
    """Return repo-relative POSIX file path -> YYYY-MM-DD last git commit date."""
    result = subprocess.run(
        ["git", "log", "--format=__COMMIT__%cs", "--name-only", "--", "docs"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}

    dates: dict[str, str] = {}
    current_date = ""
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("__COMMIT__"):
            current_date = line.removeprefix("__COMMIT__").strip()
            continue
        if current_date and line not in dates:
            dates[line] = current_date

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "docs"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty.returncode == 0:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for raw_line in dirty.stdout.splitlines():
            if len(raw_line) > 3:
                rel_path = raw_line[3:].strip().strip('"')
                if " -> " in rel_path:
                    rel_path = rel_path.split(" -> ", 1)[1].strip().strip('"')
                dates[rel_path] = today

    return dates


def resolve_source_markdown(docs_dir: Path, route_slug: str) -> Path | None:
    """Resolve a URL path slug (e.g. '' or 'models/gemma-3') to its source Markdown file."""
    clean = route_slug.strip("/")
    if not clean:
        candidate = docs_dir / "index.md"
        return candidate if candidate.is_file() else None
    for candidate in (docs_dir / f"{clean}.md", docs_dir / clean / "index.md"):
        if candidate.is_file():
            return candidate
    return None


def inject_sitemap_lastmod(
    site_dir: Path,
    docs_dir: Path,
    site_url: str | None = None,
    git_dates: dict[str, str] | None = None,
) -> int:
    """Inject <lastmod>YYYY-MM-DD</lastmod> into site/sitemap.xml entries."""
    sitemap_path = site_dir / "sitemap.xml"
    if not sitemap_path.is_file():
        return 0

    base_url = (site_url or agent_docs.SITE_URL).rstrip("/")
    repo_root = docs_dir.parent
    dates = git_dates if git_dates is not None else collect_git_lastmod_dates(repo_root)
    updated_count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal updated_count
        prefix, loc, suffix = match.group(1), match.group(2).strip(), match.group(3)
        if not loc.startswith(base_url):
            return match.group(0)
        slug = loc[len(base_url) :].strip("/")
        md_path = resolve_source_markdown(docs_dir, slug)
        if md_path is None:
            return match.group(0)
        rel_posix = md_path.relative_to(repo_root).as_posix()
        lastmod = dates.get(rel_posix)
        if not lastmod:
            lastmod = datetime.fromtimestamp(md_path.stat().st_mtime, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        updated_count += 1
        return f"{prefix}\n        <lastmod>{lastmod}</lastmod>{suffix}"

    original = sitemap_path.read_text(encoding="utf-8")
    rewritten = SITEMAP_URL_BLOCK_RE.sub(repl, original)
    if rewritten != original:
        sitemap_path.write_text(rewritten, encoding="utf-8")
    return updated_count


def inject_article_date_modified(
    site_dir: Path,
    docs_dir: Path,
    git_dates: dict[str, str] | None = None,
) -> int:
    """Inject dateModified into TechArticle JSON-LD across built HTML pages."""
    repo_root = docs_dir.parent
    dates = git_dates if git_dates is not None else collect_git_lastmod_dates(repo_root)
    marker = '{"@type":"TechArticle",'
    updated_count = 0

    for html_path in sorted(site_dir.rglob("index.html")):
        rel_dir = html_path.parent.relative_to(site_dir).as_posix()
        slug = "" if rel_dir == "." else rel_dir
        md_path = resolve_source_markdown(docs_dir, slug)
        if md_path is None:
            continue
        html = html_path.read_text(encoding="utf-8")
        if marker not in html or '"dateModified":' in html:
            continue
        rel_posix = md_path.relative_to(repo_root).as_posix()
        lastmod = dates.get(rel_posix)
        if not lastmod:
            lastmod = datetime.fromtimestamp(md_path.stat().st_mtime, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        html_path.write_text(
            html.replace(marker, f'{marker}"dateModified":"{lastmod}",', 1),
            encoding="utf-8",
        )
        updated_count += 1
    return updated_count


@contextmanager
def build_config(
    measurement_id: str | None,
    site_url: str | None = None,
) -> Iterator[Path | None]:
    """Yield a temporary Zensical config path with optional GA4 and site_url overrides."""
    if measurement_id is None and site_url is None:
        yield None
        return

    source_config = agent_docs.REPO_ROOT / "zensical.toml"
    temp_config = agent_docs.REPO_ROOT / ".zensical.build.toml"
    content = source_config.read_text(encoding="utf-8")
    if measurement_id is not None:
        content = inject_analytics_property(content, measurement_id)
    if site_url is not None:
        content = inject_site_url(content, site_url)
    temp_config.write_text(content, encoding="utf-8")
    try:
        yield temp_config
    finally:
        temp_config.unlink(missing_ok=True)


@contextmanager
def hidden_docs() -> Iterator[None]:
    """Temporarily move hidden documentation directories out of docs/ during the Zensical build."""
    moved: list[tuple[Path, Path]] = []
    stash_root = agent_docs.REPO_ROOT / ".zensical.hidden"
    try:
        for prefix in agent_docs.HIDDEN_PREFIXES:
            rel = prefix.strip("/")
            source = agent_docs.DOCS_DIR / rel
            if source.exists():
                target = stash_root / rel
                if target.exists():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
                moved.append((source, target))
        yield
    finally:
        for source, target in reversed(moved):
            if target.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                target.rename(source)
        if stash_root.exists():
            shutil.rmtree(stash_root, ignore_errors=True)


def main() -> int:
    agent_docs.write_generated()
    raw_measurement_id = os.environ.get("GA_MEASUREMENT_ID")
    measurement_id = raw_measurement_id.strip() if raw_measurement_id is not None else None
    raw_site_url = os.environ.get("SITE_URL")
    site_url = raw_site_url.strip() if raw_site_url is not None and raw_site_url.strip() else None
    is_non_prod_site = site_url is not None and site_url.rstrip("/") != "https://gemma.ai"
    if is_non_prod_site:
        measurement_id = ""
    raw_colab_repo = os.environ.get("COLAB_REPO")
    colab_repo = raw_colab_repo.strip() if raw_colab_repo is not None and raw_colab_repo.strip() else None
    raw_colab_branch = os.environ.get("COLAB_BRANCH")
    colab_branch = (
        raw_colab_branch.strip() if raw_colab_branch is not None and raw_colab_branch.strip() else None
    )
    args = sys.argv[1:]
    site_dir = agent_docs.REPO_ROOT / "site"
    for prefix in agent_docs.HIDDEN_PREFIXES:
        stale = site_dir / prefix.strip("/")
        if stale.exists():
            shutil.rmtree(stale)
    with hidden_docs(), build_config(measurement_id, site_url) as temp_config:
        if temp_config and "-f" not in args and "--config-file" not in args:
            args = ["-f", str(temp_config), *args]
        command = [sys.executable, "-m", "zensical", "build", *args]
        result = subprocess.run(command, cwd=agent_docs.REPO_ROOT, check=False)
        if result.returncode:
            return result.returncode
        if is_non_prod_site and colab_repo and colab_branch:
            rewrite_colab_links(site_dir, colab_repo, colab_branch)
        agent_docs.publish(site_dir)
    git_dates = collect_git_lastmod_dates(agent_docs.REPO_ROOT)
    inject_sitemap_lastmod(site_dir, agent_docs.DOCS_DIR, site_url, git_dates=git_dates)
    inject_article_date_modified(site_dir, agent_docs.DOCS_DIR, git_dates=git_dates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
