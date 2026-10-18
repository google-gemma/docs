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

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_docs

HAS_ZENSICAL = importlib.util.find_spec("zensical") is not None


class AnalyticsBuildTests(unittest.TestCase):
    def test_inject_analytics_property_updates_config(self):
        source = (build_docs.agent_docs.REPO_ROOT / "zensical.toml").read_text(encoding="utf-8")
        self.assertIn('property = "G-CWMC2KTTGQ"', source)
        injected = build_docs.inject_analytics_property(source, "G-TEST123456")
        self.assertIn('property = "G-TEST123456"', injected)
        cleared = build_docs.inject_analytics_property(source, "")
        self.assertIn('property = ""', cleared)

    @unittest.skipUnless(HAS_ZENSICAL, "zensical is required for full build tests")
    def test_build_without_measurement_id_omits_tracking_script(self):
        env = {**os.environ, "GA_MEASUREMENT_ID": ""}
        result = subprocess.run(
            [sys.executable, "scripts/build_docs.py"],
            cwd=build_docs.agent_docs.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        index_html = (build_docs.agent_docs.REPO_ROOT / "site" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("googletagmanager.com/gtag/js", index_html)
        self.assertNotIn("Change cookie settings", index_html)

    @unittest.skipUnless(HAS_ZENSICAL, "zensical is required for full build tests")
    def test_build_with_measurement_id_includes_ga4_and_consent(self):
        env = {**os.environ, "GA_MEASUREMENT_ID": "G-TEST123456"}
        result = subprocess.run(
            [sys.executable, "scripts/build_docs.py"],
            cwd=build_docs.agent_docs.REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        index_html = (build_docs.agent_docs.REPO_ROOT / "site" / "index.html").read_text(encoding="utf-8")
        self.assertIn("https://www.googletagmanager.com/gtag/js?id=G-TEST123456", index_html)
        self.assertIn("G-TEST123456", index_html)
        self.assertIn('id="__consent"', index_html)
        self.assertIn('name="analytics"', index_html)
        self.assertIn("Change cookie settings", index_html)
        # Ensure temporary config file is cleaned up
        self.assertFalse((build_docs.agent_docs.REPO_ROOT / ".zensical.build.toml").exists())


if __name__ == "__main__":
    unittest.main()
