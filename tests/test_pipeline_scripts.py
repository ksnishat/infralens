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


class TestMetricNameSanitization:
    """MLflow rejects metric names containing parentheses.

    YOLO reports keys such as "metrics/precision(B)", so logging them raw fails
    the whole training task after the model has already trained.
    """

    def test_parentheses_are_removed(self):
        from src.scripts.train_yolo import _sanitize_metric_name

        assert _sanitize_metric_name("metrics/precision(B)") == "metrics/precision_B"
        assert _sanitize_metric_name("metrics/mAP50-95(B)") == "metrics/mAP50-95_B"

    def test_plain_names_are_unchanged(self):
        from src.scripts.train_yolo import _sanitize_metric_name

        assert _sanitize_metric_name("fitness") == "fitness"
        assert _sanitize_metric_name("train/box_loss") == "train/box_loss"

    def test_result_is_mlflow_safe(self):
        import re

        from src.scripts.train_yolo import _sanitize_metric_name

        allowed = re.compile(r"^[A-Za-z0-9_\-. :/]+$")
        for key in [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "val/box_loss",
        ]:
            assert allowed.match(_sanitize_metric_name(key)), key


class TestTrainingCli:
    """The training script must be runnable without arguments.

    train_yolo() previously defaulted data_yaml to "data/gold/yolo_dataset.yaml"
    and __main__ called train_yolo() with no arguments, so
    `python src/scripts/train_yolo.py` failed on any host checkout. There were
    no flags to override it either.
    """

    def test_default_data_yaml_resolves_on_a_host_checkout(self):
        from src.scripts.train_yolo import _default_data_yaml

        path = _default_data_yaml()
        assert os.path.exists(path), f"resolved to a missing path: {path}"
        assert path.endswith("data.yaml")

    def test_default_data_yaml_prefers_the_env_override(self, tmp_path, monkeypatch):
        from src.scripts.train_yolo import _default_data_yaml

        fake = tmp_path / "data.yaml"
        fake.write_text("nc: 1\n")
        monkeypatch.setenv("INFRALENS_DATA_YAML", str(fake))
        assert _default_data_yaml() == str(fake)

    def test_train_yolo_resolves_the_path_instead_of_hardcoding_it(self):
        """The default must be None so resolution happens at call time."""
        import inspect

        from src.scripts.train_yolo import train_yolo

        default = inspect.signature(train_yolo).parameters["data_yaml"].default
        assert default is None, (
            f"data_yaml defaults to {default!r}; it must be None so "
            "_default_data_yaml() is consulted"
        )

    def test_cli_exposes_every_train_parameter(self):
        """A parameter with no flag can only be changed by editing the source."""
        import inspect
        import subprocess
        import sys

        from src.scripts.train_yolo import train_yolo

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, "src/scripts/train_yolo.py", "--help"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        assert result.returncode == 0, result.stderr
        help_text = result.stdout

        params = set(inspect.signature(train_yolo).parameters)
        expected = {p.replace("_", "-") for p in params}
        missing = {p for p in expected if f"--{p}" not in help_text}
        assert not missing, f"train_yolo() parameters with no CLI flag: {sorted(missing)}"
