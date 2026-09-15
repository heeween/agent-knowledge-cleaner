from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.sync_release import remote_preflight, require_remote_root, remote_rollback, stage_and_switch


class SyncReleaseTest(unittest.TestCase):
    def test_remote_root_scope_guard(self):
        self.assertEqual(require_remote_root("/root/yj-kb/cleaner_releases"), "/root/yj-kb/cleaner_releases")
        for value in ("/", "/root", "/tmp/kb", "/root/yj-kb/../other"):
            with self.assertRaises(ValueError):
                require_remote_root(value)

    @patch("scripts.sync_release.run_ssh")
    def test_read_only_preflight(self, ssh):
        ssh.return_value = subprocess.CompletedProcess([], 0, b"ABSENT\n1.0.0\n", b"")
        result = remote_preflight("example.invalid", "/root/yj-kb/kb_releases", "1.0.1")
        self.assertFalse(result["release_present"])
        self.assertEqual(ssh.call_count, 1)
        self.assertNotIn("mkdir", ssh.call_args.args[1])

    @patch("scripts.sync_release.run_ssh")
    def test_rollback_preflight_does_not_switch(self, ssh):
        ssh.return_value = subprocess.CompletedProcess([], 0, b"chunks.jsonl: OK\n1.0.1\n", b"")
        result = remote_rollback("example.invalid", "/root/yj-kb/kb_releases", "1.0.0", 5006, apply=False)
        self.assertTrue(result["dry_run"])
        self.assertEqual(ssh.call_count, 1)
        self.assertNotIn("ln -s", ssh.call_args.args[1])

    @patch("scripts.sync_release.run_ssh")
    def test_hot_switch_and_explicit_rollback_command_flow(self, ssh):
        outputs = [b"200", b"", b"", b"chunks.jsonl: OK\n", b"", b"1.0.0\n", b"", b"200"]
        ssh.side_effect = [subprocess.CompletedProcess([], 0, output, b"") for output in outputs]
        release = Path(__file__).resolve().parents[1] / "releases" / "1.0.1"
        result = stage_and_switch("example.invalid", "/root/yj-kb/cleaner_releases", "1.0.1", release, 5006)
        self.assertEqual(result["previous"], "1.0.0")
        self.assertEqual(result["reload_http"], "200")
        commands = [call.args[1] for call in ssh.call_args_list]
        self.assertTrue(any("sha256sum -c SHA256SUMS" in cmd for cmd in commands))
        self.assertTrue(any("/kb/reload" in cmd for cmd in commands))

        ssh.reset_mock()
        ssh.side_effect = [
            subprocess.CompletedProcess([], 0, b"chunks.jsonl: OK\n1.0.1\n", b""),
            subprocess.CompletedProcess([], 0, b"200", b""),
            subprocess.CompletedProcess([], 0, b"1.0.1\n", b""),
            subprocess.CompletedProcess([], 0, b"200", b""),
        ]
        rolled = remote_rollback("example.invalid", "/root/yj-kb/cleaner_releases", "1.0.0", 5006, apply=True)
        self.assertEqual(rolled["rollback_to"], "1.0.0")
        self.assertEqual(rolled["reload_http"], "200")


if __name__ == "__main__":
    unittest.main()
