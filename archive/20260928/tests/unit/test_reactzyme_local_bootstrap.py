"""Test bootstrap quoting and failure behavior without tmux, NFS, or GPUs."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/launch_reactzyme_prott5_local_bootstrap.sh"


class LocalBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.env = {**os.environ, "PATH": str(self.bin) + ":/usr/bin:/bin",
                    "REACTZYME_TEST_LOGDIR": str(self.logs),
                    "REACTZYME_TEST_HOST": "slurm-node-014"}
        self.command("hostname", 'echo "$REACTZYME_TEST_HOST"')
        self.command("mktemp", 'echo "$REACTZYME_TEST_LOGDIR"')
        self.command("timeout", "exit 124")
        # Execute the payload synchronously, but return tmux's startup-success
        # status even if its child fails. The failure path must never run Python.
        self.command("tmux", '''
if [ "$1" = has-session ]; then
  exit "${REACTZYME_TEST_SESSION_STATUS:-1}"
fi
[ "$1" = new-session ] && [ "$5" = -c ] && [ "$6" = / ] || exit 70
[ "$7" = /usr/bin/env ] || exit 71
shift 6
"$@" || true
''')

    def command(self, name, text):
        path = self.bin / name
        path.write_text("#!/bin/sh\n" + text + "\n")
        path.chmod(0o700)

    def tearDown(self):
        self.temp.cleanup()

    def launch(self):
        return subprocess.run(["/bin/bash", "--noprofile", "--norc", str(SCRIPT)],
                              env=self.env, cwd="/", text=True, capture_output=True, timeout=5)

    def test_controller_timeout_is_logged_locally_without_starting_extraction(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        log = (self.logs / "bootstrap.log").read_text()
        self.assertIn("Local bootstrap started", log)
        self.assertIn("exit 124", log)
        self.assertIn("No extraction started", log)
        self.assertNotIn("Controller is readable", log)
        self.assertIn(str(self.logs / "bootstrap.log"), result.stdout)

    def test_wrong_node_refuses_before_creating_a_log(self):
        self.env["REACTZYME_TEST_HOST"] = "slurm-node-013"
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Nothing was launched", result.stdout)
        self.assertFalse((self.logs / "bootstrap.log").exists())

    def test_existing_session_is_never_replaced(self):
        self.env["REACTZYME_TEST_SESSION_STATUS"] = "0"
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Not launching another copy", result.stdout)
        self.assertFalse((self.logs / "bootstrap.log").exists())

    def test_local_log_path_with_spaces(self):
        self.logs = self.root / "log path with spaces"
        self.logs.mkdir()
        self.env["REACTZYME_TEST_LOGDIR"] = str(self.logs)
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("exit 124", (self.logs / "bootstrap.log").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
