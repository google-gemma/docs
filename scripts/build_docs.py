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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
