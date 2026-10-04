"""Integration tests for neurofuck CLI commands."""

import unittest
import tempfile
import subprocess
import sys
from pathlib import Path


class TestCLI(unittest.TestCase):
    """Test CLI commands execution."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_pipeline(self):
        model_file = self.dir_path / "model.json"
        bf_file = self.dir_path / "model.bf"

        # 1. Train
        res = subprocess.run(
            [sys.executable, "-m", "neurofuck.cli", "train", "--dataset", "and", "--output", str(model_file), "--epochs", "2000", "--lr", "0.1"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Train failed: {res.stderr}")
        self.assertTrue(model_file.exists())

        # 2. Compile
        res = subprocess.run(
            [sys.executable, "-m", "neurofuck.cli", "compile", str(model_file), "--output", str(bf_file), "--scale", "16"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Compile failed: {res.stderr}")
        self.assertTrue(bf_file.exists())

        # 3. Verify
        res = subprocess.run(
            [sys.executable, "-m", "neurofuck.cli", "verify", str(model_file), "--dataset", "and", "--scale", "16"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Verify failed: {res.stderr}")
        self.assertIn("VERIFICATION PASSED", res.stdout)

        # 4. Run single sample
        res = subprocess.run(
            [sys.executable, "-m", "neurofuck.cli", "run", str(bf_file), "--input", "1.0,1.0", "--scale", "16"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Run failed: {res.stderr}")
        self.assertIn("Outputs (float):", res.stdout)


if __name__ == "__main__":
    unittest.main()
