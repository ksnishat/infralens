"""Tests for the DVC pipeline scripts.

extract_bronze.py previously contained a Python 3.12-only f-string that would
not parse on 3.10, and it never actually copied any files.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT = REPO_ROOT / "src" / "scripts" / "extract_bronze.py"


def _make_image(path: Path, size=(32, 32), color="gray"):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="JPEG")


def _run(script: Path, *args):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


class TestExtractBronze:
    def test_script_parses_on_this_interpreter(self):
        """Regression: the file contained a 3.12-only f-string and would not
        even compile on 3.10."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(EXTRACT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_copies_files_and_preserves_folders(self, tmp_path):
        """Regression: the script logged filenames but never copied anything."""
        src = tmp_path / "raw"
        _make_image(src / "a" / "x.jpg")
        _make_image(src / "b" / "x.jpg")
        out = tmp_path / "bronze"

        result = _run(EXTRACT, "--source", str(src), "--output", str(out))

        assert result.returncode == 0, result.stderr
        assert (out / "a" / "x.jpg").exists()
        assert (out / "b" / "x.jpg").exists()

    def test_reports_the_number_copied(self, tmp_path):
        src = tmp_path / "raw"
        _make_image(src / "a" / "x.jpg")
        _make_image(src / "a" / "y.jpg")
        out = tmp_path / "bronze"

        result = _run(EXTRACT, "--source", str(src), "--output", str(out))

        assert "2 files copied" in result.stdout

    def test_missing_source_raises(self, tmp_path):
        result = _run(
            EXTRACT, "--source", str(tmp_path / "nope"), "--output", str(tmp_path / "out")
        )
        assert result.returncode != 0
        assert "Source directory not found" in result.stderr
