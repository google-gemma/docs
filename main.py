"""Macros available to every page, loaded automatically by the macros extension.

Currently one: `icon()`, which inlines an SVG from the icon sets that ship with Zensical.

Why a macro rather than the `:lucide-rocket:` shortcode. The shortcode is an *inline* Markdown
construct, so it renders in prose but stays literal text inside a raw HTML block -- and every
card grid on this site is a raw HTML block. Marking those blocks `markdown` does render it, but
`md_in_html` then wraps each card in a `<p>`, which breaks the grid layout.

This macro runs in the Jinja preprocessor, before Markdown sees the page, so the SVG is already
in place by the time the HTML block is passed through untouched. It also works in the grids that
are generated from `data/*.yml`, where a shortcode could not reach at all.

The shortcode is still the right tool in ordinary prose. Use this one inside markup.
"""

from __future__ import annotations

import importlib.util
import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Icon names are used to build a file path, so keep them to a strict shape.
VENDOR_ICONS = REPO_ROOT / "overrides" / ".icons"
ICON_NAME = re.compile(r"^[a-z0-9]+(?:/[a-z0-9-]+)+$")

# Sized by CSS, not by the file: the shipped sets disagree about their intrinsic dimensions.
SIZING_ATTR = re.compile(r'\s(?:width|height)="[^"]*"')
SVG_OPEN = re.compile(r"<svg\b")
# The shipped sets already carry a class of their own (`lucide lucide-rocket`). Merge into it
# rather than adding a second class attribute, which is invalid -- browsers keep only the first.
SVG_CLASS = re.compile(r'(<svg\b[^>]*?)\sclass="([^"]*)"')


@lru_cache(maxsize=1)
def _icon_roots() -> tuple[Path, ...]:
    """Where to look for icons: the project's own overrides first, then Zensical's sets."""
    roots = [VENDOR_ICONS]
    spec = importlib.util.find_spec("zensical")
    if spec and spec.origin:
        roots.append(Path(spec.origin).parent / "templates" / ".icons")
    return tuple(root for root in roots if root.is_dir())


@lru_cache(maxsize=512)
def _load_icon(name: str, css_class: str) -> str:
    if not ICON_NAME.match(name):
        raise ValueError(f"icon({name!r}): expected a name like 'lucide/rocket'")

    for root in _icon_roots():
        path = root.joinpath(*name.split("/")).with_suffix(".svg")
        if path.is_file():
            svg = SIZING_ATTR.sub("", path.read_text(encoding="utf-8").strip())
            if SVG_CLASS.search(svg):
                svg = SVG_CLASS.sub(rf'\g<1> class="\g<2> {css_class}"', svg, count=1)
            else:
                svg = SVG_OPEN.sub(f'<svg class="{css_class}"', svg, count=1)
            return SVG_OPEN.sub('<svg aria-hidden="true"', svg, count=1)

    if not _icon_roots():
        # No roots at all means Zensical isn't installed and nothing is vendored -- the lint
        # workflow's environment. Name the fix rather than reporting an empty set list.
        raise ValueError(
            f"icon({name!r}): no icon sets available. Zensical is not installed and "
            f"{VENDOR_ICONS.relative_to(REPO_ROOT).as_posix()}/ is empty -- "
            "run `uv run python scripts/vendor_icons.py` and commit the result."
        )
    sets = ", ".join(sorted(p.name for root in _icon_roots() for p in root.iterdir() if p.is_dir()))
    raise ValueError(
        f"icon({name!r}): not found in {sets}. If it exists in Zensical, "
        "run `uv run python scripts/vendor_icons.py` to vendor it."
    )


def icon(name: str, css_class: str = "gemma-icon") -> str:
    """Inline an icon set SVG, e.g. `{{ icon('lucide/rocket') }}`.

    Sets that ship with Zensical: lucide, material, fontawesome, octicons, simple.
    `simple` holds brand logos. Project-specific icons go in `overrides/.icons/`.

    The SVG carries `stroke="currentColor"` or `fill="currentColor"`, so it takes the colour of
    its surroundings; size it with CSS on `.gemma-icon`.

    Importable so `scripts/agent_docs.py` can offer the same macro when it renders the raw
    Markdown routes -- those go through their own Jinja environment.
    """
    return _load_icon(name, css_class)


def define_env(env) -> None:
    env.macro(icon)
