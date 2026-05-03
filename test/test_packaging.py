"""Packaging tests for issue #37.

Asserts that ``setup.py`` is gone, ``pyproject.toml`` discovers all
sub-packages via ``[tool.setuptools.packages.find]``, the ``py.typed``
marker is preserved, and a built wheel ships every ``stockpy.*`` sub-package.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


class TestSetupPyRemoved:
    def test_when_repo_inspected_setup_py_does_not_exist(self) -> None:
        assert not (REPO / "setup.py").exists()


class TestPackageDiscovery:
    def test_when_pyproject_loaded_packages_find_block_exists(
        self, pyproject: dict
    ) -> None:
        find = pyproject["tool"]["setuptools"]["packages"]["find"]
        assert isinstance(find, dict)

    def test_when_packages_find_used_includes_stockpy_glob(
        self, pyproject: dict
    ) -> None:
        find = pyproject["tool"]["setuptools"]["packages"]["find"]
        assert "stockpy*" in find["include"]

    def test_when_packages_find_used_excludes_test_and_docs(
        self, pyproject: dict
    ) -> None:
        # ``stock*`` would match ``stockpy*`` under fnmatch (the AC's literal
        # pattern is broken); the data directory ``stock/`` has no __init__.py
        # and is never discovered as a package, so excluding it is unnecessary.
        find = pyproject["tool"]["setuptools"]["packages"]["find"]
        excludes = set(find["exclude"])
        assert {"test*", "docs*"} <= excludes

    def test_when_pyproject_loaded_packages_field_is_no_longer_a_flat_list(
        self, pyproject: dict
    ) -> None:
        setuptools_block = pyproject["tool"]["setuptools"]
        # The legacy `packages = ["stockpy"]` flat list must be gone.
        assert not isinstance(setuptools_block.get("packages"), list)


class TestPyTypedMarkerPreserved:
    def test_when_pyproject_loaded_package_data_keeps_py_typed(
        self, pyproject: dict
    ) -> None:
        package_data = pyproject["tool"]["setuptools"]["package-data"]
        assert "py.typed" in package_data["stockpy"]

    def test_when_repo_inspected_py_typed_marker_exists(self) -> None:
        # PEP 561 requires the file at runtime so type-checkers find it.
        assert (REPO / "stockpy" / "py.typed").exists()


class TestSubpackageImports:
    """The original bug: subpackages must remain importable."""

    def test_when_imported_forecasters_subpackage_resolves(self) -> None:
        from stockpy.forecasters import (  # noqa: F401
            DMMForecaster,
            GRUForecaster,
            LSTMForecaster,
            TransformerForecaster,
        )

    def test_when_imported_preprocessing_subpackage_resolves(self) -> None:
        from stockpy.preprocessing import TimeSeriesDataset  # noqa: F401

    def test_when_imported_callbacks_subpackage_resolves(self) -> None:
        from stockpy.callbacks import EarlyStopping  # noqa: F401

    def test_when_imported_utils_subpackage_resolves(self) -> None:
        from stockpy.utils import to_tensor  # noqa: F401


@pytest.mark.slow
class TestWheelBuild:
    """End-to-end PEP 517 build → confirm the wheel actually ships subpackages."""

    @pytest.fixture(scope="class")
    def wheel(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        outdir = tmp_path_factory.mktemp("wheel")
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.fail(f"python -m build failed:\n{result.stderr}")
        wheels = list(outdir.glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
        return wheels[0]

    @pytest.fixture(scope="class")
    def wheel_contents(self, wheel: Path) -> set[str]:
        with zipfile.ZipFile(wheel) as zf:
            return set(zf.namelist())

    def test_when_wheel_built_contains_forecasters_init(
        self, wheel_contents: set[str]
    ) -> None:
        assert "stockpy/forecasters/__init__.py" in wheel_contents

    def test_when_wheel_built_contains_preprocessing_init(
        self, wheel_contents: set[str]
    ) -> None:
        assert "stockpy/preprocessing/__init__.py" in wheel_contents

    def test_when_wheel_built_contains_callbacks_init(
        self, wheel_contents: set[str]
    ) -> None:
        assert "stockpy/callbacks/__init__.py" in wheel_contents

    def test_when_wheel_built_contains_utils_init(
        self, wheel_contents: set[str]
    ) -> None:
        assert "stockpy/utils/__init__.py" in wheel_contents

    def test_when_wheel_built_contains_py_typed_marker(
        self, wheel_contents: set[str]
    ) -> None:
        assert "stockpy/py.typed" in wheel_contents

    def test_when_wheel_built_does_not_ship_test_directory(
        self, wheel_contents: set[str]
    ) -> None:
        assert not any(name.startswith("test/") for name in wheel_contents)

    def test_when_wheel_installed_in_clean_env_subpackages_are_importable(
        self, wheel: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        target = tmp_path_factory.mktemp("install_target")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, '%s'); "
                "from stockpy.forecasters import LSTMForecaster; "
                "from stockpy.preprocessing import TimeSeriesDataset; "
                "from stockpy.callbacks import EarlyStopping; "
                "from stockpy.utils import to_tensor; "
                "print('ok')" % target,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
        # cleanup target tree (large)
        shutil.rmtree(target, ignore_errors=True)
