"""Annotation-coverage tests for ``stockpy.forecasters``.

Coverage for issue #35: every file under ``stockpy/forecasters/`` must start
with ``from __future__ import annotations`` and every public class must have
fully-annotated ``__init__`` / ``forward`` (and equivalent) signatures.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import stockpy.forecasters as forecasters_pkg

FORECASTERS_DIR = Path(forecasters_pkg.__file__).parent
PUBLIC_METHODS = ("__init__", "forward", "initialize_module")


def _forecaster_files() -> list[Path]:
    return sorted(p for p in FORECASTERS_DIR.glob("*.py") if p.name != "__init__.py")


def _forecaster_modules() -> list[str]:
    return [
        info.name
        for info in pkgutil.iter_modules([str(FORECASTERS_DIR)])
        if info.name.startswith("_") and info.name != "__init__"
    ]


def _first_real_statement(tree: ast.Module) -> ast.stmt | None:
    """Return the first statement skipping the optional module docstring."""
    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1] if len(body) > 1 else None
    return body[0] if body else None


def _has_full_annotations(func: object) -> bool:
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue  # **kwargs is conventionally untyped
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue  # *args same
        if param.annotation is inspect.Parameter.empty:
            return False
    return sig.return_annotation is not inspect.Signature.empty


@pytest.mark.parametrize("path", _forecaster_files(), ids=lambda p: p.name)
class TestFutureAnnotationsImport:
    def test_when_parsed_first_real_statement_is_future_annotations(
        self, path: Path
    ) -> None:
        tree = ast.parse(path.read_text())
        first = _first_real_statement(tree)
        assert isinstance(first, ast.ImportFrom), (
            f"{path.name}: first statement is not an import"
        )
        assert first.module == "__future__"
        assert any(alias.name == "annotations" for alias in first.names)


@pytest.mark.parametrize("module_name", _forecaster_modules())
class TestPublicMethodAnnotations:
    """Every public class's __init__, forward, initialize_module are typed."""

    def test_when_classes_introspected_public_methods_have_annotations(
        self, module_name: str
    ) -> None:
        module = importlib.import_module(f"stockpy.forecasters.{module_name}")
        public_classes = [
            cls
            for name, cls in inspect.getmembers(module, inspect.isclass)
            if cls.__module__ == module.__name__ and not name.startswith("_")
        ]
        assert public_classes, (
            f"{module_name}: no public classes discovered for annotation check"
        )
        for cls in public_classes:
            for method_name in PUBLIC_METHODS:
                func = cls.__dict__.get(method_name)
                if func is None:
                    continue
                assert _has_full_annotations(func), (
                    f"{cls.__name__}.{method_name} is missing annotations"
                )


class TestImportDoesNotBreak:
    def test_when_stockpy_forecasters_imported_no_circular_import(self) -> None:
        # importing the package and each submodule must not raise.
        for module_name in _forecaster_modules():
            importlib.import_module(f"stockpy.forecasters.{module_name}")
        importlib.import_module("stockpy")
