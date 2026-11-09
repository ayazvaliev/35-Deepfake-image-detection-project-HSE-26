import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "src" / "notebook_src"


class NotebookImportTests(unittest.TestCase):
    def run_python(self, source):
        with tempfile.TemporaryDirectory() as temporary:
            environment = os.environ.copy()
            environment["DEEPFAKE_DATA_ROOT"] = str(Path(temporary) / "absent_data")
            environment["MPLBACKEND"] = "Agg"
            result = subprocess.run(
                [sys.executable, "-c", source],
                cwd=ROOT / "notebooks",
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_helpers_import_as_package_without_reading_dataset(self):
        names = sorted(path.stem for path in HELPERS.glob("*.py") if path.stem != "__init__")
        source = f"""
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, {str(ROOT)!r})
original_rglob = Path.rglob
dataset = Path(os.environ["DEEPFAKE_DATA_ROOT"]) / "openforensics"

def checked_rglob(path, pattern):
    if path == dataset:
        raise AssertionError("Dataset discovery ran during import")
    return original_rglob(path, pattern)

with patch.object(Path, "rglob", checked_rglob):
    modules = [importlib.import_module("src.notebook_src." + name) for name in {names!r}]
assert len(modules) == {len(names)}
assert not any(name.startswith("of_") for name in sys.modules)
"""
        self.run_python(source)

    def test_raw_paths_match_repository_root_for_relative_absolute_and_home_paths(self):
        source = f"""
import importlib
import json
import os
import sys
from pathlib import Path

root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
notebook = json.loads((root / "notebooks/eda_mffi.ipynb").read_text())
cell = "".join(notebook["cells"][4]["source"])
prefix = cell.split("label_files =", 1)[0]
for raw in ["data/raw", "~/deepfake-path-check", os.environ["DEEPFAKE_DATA_ROOT"]]:
    os.environ["DEEPFAKE_DATA_ROOT"] = raw
    expected = Path(raw).expanduser()
    if not expected.is_absolute():
        expected = root / expected
    config = importlib.reload(importlib.import_module("src.notebook_src.config"))
    of_data = importlib.reload(importlib.import_module("src.notebook_src.of_data"))
    namespace = {{}}
    exec(prefix, namespace)
    assert config.RAW_DIR == expected, (config.RAW_DIR, expected)
    assert of_data.RAW == expected, (of_data.RAW, expected)
    assert namespace["RAW"] == expected, (namespace["RAW"], expected)
    assert namespace["MFFI"] == expected / "mffi"
"""
        self.run_python(source)

    def test_direct_entry_script_imports_work_from_notebooks(self):
        entries = [str(HELPERS / f"of_{name}.py") for name in ["scan", "unlabeled", "attributes"]]
        source = f"""
import runpy
import sys

for index, script in enumerate({entries!r}):
    sys.argv = [script, "Train"]
    namespace = runpy.run_path(script, run_name="import_check_" + str(index))
    assert callable(namespace["scan"])
assert "src.notebook_src.of_data" in sys.modules
assert not any(name.startswith("of_") for name in sys.modules)
"""
        self.run_python(source)


if __name__ == "__main__":
    unittest.main()
