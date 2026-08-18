"""Report WCAG contrast for the brand blue ramp.

The ramp in `docs/stylesheets/gemma.css` is derived from a single seed with `color-mix`,
so changing the seed changes every step. This script answers the only question that
matters when you do: does the new seed still pass?

    uv run python scripts/check_contrast.py                  # check the seed in gemma.css
    uv run python scripts/check_contrast.py --seed "#0B57D0" # try one before committing

Exits non-zero if a step misses the threshold its role requires, so it can gate a change
without anyone having to eyeball a swatch.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GEMMA_CSS = REPO_ROOT / "docs" / "stylesheets" / "gemma.css"

SEED_DECL = re.compile(r"--gemma-blue-seed:\s*(#[0-9A-Fa-f]{6})\s*;")
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

WHITE = "#FFFFFF"
INK = "#202124"

# step -> (mix percentage, colour mixed with, role, what it must clear)
#
# "white-on" means white text sits on this step; "on-ink" means the step is used as text on a
# dark surface. 500 is decorative only -- gradients and 2px rules, which carry no text -- so it
# has no threshold. Keep this table in step with the ramp in gemma.css.
RAMP: tuple[tuple[str, int, str, str, str | None, float | None], ...] = (
    ("50", 8, WHITE, "tints, hover washes", None, None),
    ("100", 20, WHITE, "tints", None, None),
    ("300", 52, WHITE, "dark-scheme text", "on-ink", 4.5),
    ("500", 78, WHITE, "decorative only", None, None),
    ("600", 90, WHITE, "dark-scheme fills", "white-on", 4.5),
    ("700", 100, WHITE, "primary: accent + light fills", "both", 4.5),
    ("800", 78, "#000000", "pressed / strong", "white-on", 4.5),
)


def _channels(colour: str) -> tuple[int, int, int]:
    colour = colour.lstrip("#")
    return tuple(int(colour[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def mix(seed: str, pct: int, other: str) -> str:
    """Replicate `color-mix(in srgb, seed pct%, other)` -- linear interpolation in gamma space."""
    if pct == 100:
        return seed.upper()
    s, o = _channels(seed), _channels(other)
    f = pct / 100
    return "#%02X%02X%02X" % tuple(round(s[i] * f + o[i] * (1 - f)) for i in range(3))


def luminance(colour: str) -> float:
    def channel(value: int) -> float:
        v = value / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = _channels(colour)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def seed_from_css() -> str:
    match = SEED_DECL.search(GEMMA_CSS.read_text(encoding="utf-8"))
    if not match:
        sys.exit(f"no --gemma-blue-seed declaration found in {GEMMA_CSS}")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", help="try a seed without editing gemma.css, e.g. '#0B57D0'")
    args = parser.parse_args()

    seed = args.seed or seed_from_css()
    if not HEX.match(seed):
        sys.exit(f"seed must be a 6-digit hex colour like '#0B57D0', got {seed!r}")
    seed = seed.upper()

    origin = "--seed" if args.seed else "gemma.css"
    print(f"seed {seed}  (from {origin})\n")
    print(f"{'step':6} {'hex':9} {'white-on':>9} {'on-white':>9} {'on-ink':>8}  role")
    print("-" * 78)

    failures: list[str] = []
    for name, pct, other, role, requirement, threshold in RAMP:
        hexv = mix(seed, pct, other)
        white_on = contrast(WHITE, hexv)
        on_ink = contrast(hexv, INK)

        flag = ""
        if requirement and threshold:
            checks = {
                "white-on": [("white text on it", white_on)],
                "on-ink": [("as text on ink", on_ink)],
                "both": [("white text on it", white_on), ("as text on white", white_on)],
            }[requirement]
            for label, value in checks:
                if value < threshold:
                    failures.append(f"{name} ({role}): {label} is {value:.2f}:1, needs {threshold}:1")
                    flag = "  <-- FAILS"

        print(f"{name:6} {hexv:9} {white_on:>9.2f} {white_on:>9.2f} {on_ink:>8.2f}  {role}{flag}")

    print()
    if failures:
        print("FAILED")
        for line in failures:
            print(f"  {line}")
        print("\nPick a darker seed: white-on rises as the seed darkens.")
        return 1

    print("All steps clear the threshold their role requires.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
