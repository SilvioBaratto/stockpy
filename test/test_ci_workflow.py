"""Structural tests for the CI workflow.

Coverage for issue #36 acceptance criteria. Asserts the GitHub Actions
workflow at ``.github/workflows/python-package.yml`` matches the v0.4.0
expected shape: branch triggers, Python matrix, dev-extras install,
``pytest --cov``, ruff + black lint, modern Coveralls action, pip caching.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/python-package.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def steps(workflow: dict) -> list[dict]:
    jobs = workflow["jobs"]
    job = next(iter(jobs.values()))
    return job["steps"]


def _step_run(steps: list[dict], substring: str) -> dict | None:
    for step in steps:
        if substring in (step.get("run") or ""):
            return step
    return None


def _step_uses(steps: list[dict], action_prefix: str) -> dict | None:
    for step in steps:
        uses = step.get("uses") or ""
        if uses.startswith(action_prefix):
            return step
    return None


class TestTriggers:
    def test_when_push_event_branches_include_main_and_development(
        self, workflow: dict
    ) -> None:
        # ``on`` parses as boolean True under PyYAML — fall back to the truthy key.
        on = workflow.get("on") or workflow.get(True)
        assert set(on["push"]["branches"]) >= {"main", "development"}

    def test_when_pull_request_event_branches_include_main_and_development(
        self, workflow: dict
    ) -> None:
        on = workflow.get("on") or workflow.get(True)
        assert set(on["pull_request"]["branches"]) >= {"main", "development"}


class TestPythonMatrix:
    def test_when_matrix_listed_python_versions_are_3_10_3_11_3_12(
        self, workflow: dict
    ) -> None:
        job = next(iter(workflow["jobs"].values()))
        versions = [str(v) for v in job["strategy"]["matrix"]["python-version"]]
        assert versions == ["3.10", "3.11", "3.12"]


class TestActionVersions:
    def test_when_checkout_action_used_version_is_v4(self, steps: list[dict]) -> None:
        step = _step_uses(steps, "actions/checkout@")
        assert step is not None
        assert step["uses"] == "actions/checkout@v4"

    def test_when_setup_python_action_used_version_is_v5(
        self, steps: list[dict]
    ) -> None:
        step = _step_uses(steps, "actions/setup-python@")
        assert step is not None
        assert step["uses"] == "actions/setup-python@v5"

    def test_when_setup_python_used_pip_cache_is_enabled(
        self, steps: list[dict]
    ) -> None:
        step = _step_uses(steps, "actions/setup-python@")
        assert step is not None
        assert step["with"].get("cache") == "pip"


class TestInstallStep:
    def test_when_install_step_runs_dev_extras_are_installed_from_pyproject(
        self, steps: list[dict]
    ) -> None:
        step = _step_run(steps, 'pip install -e ".[dev]"')
        assert step is not None, "install step must use editable [dev] extras"

    def test_when_install_step_runs_pip_is_upgraded_first(
        self, steps: list[dict]
    ) -> None:
        step = _step_run(steps, 'pip install -e ".[dev]"')
        assert step is not None
        assert "pip install --upgrade pip" in step["run"]


class TestTestStep:
    def test_when_pytest_step_runs_uses_cov_xml_and_term(
        self, steps: list[dict]
    ) -> None:
        step = _step_run(steps, "pytest")
        assert step is not None
        run = step["run"]
        assert "--cov=stockpy" in run
        assert "--cov-report=xml" in run
        assert "--cov-report=term" in run


class TestLintSteps:
    def test_when_lint_step_runs_uses_ruff_check_on_stockpy(
        self, steps: list[dict]
    ) -> None:
        assert _step_run(steps, "ruff check stockpy/") is not None

    def test_when_lint_step_runs_uses_black_check_on_stockpy(
        self, steps: list[dict]
    ) -> None:
        assert _step_run(steps, "black --check stockpy/") is not None

    def test_when_lint_steps_run_pycodestyle_is_no_longer_used(
        self, steps: list[dict]
    ) -> None:
        for step in steps:
            assert "pycodestyle" not in (step.get("run") or "")


class TestCoverallsUpload:
    def test_when_coverage_uploaded_uses_modern_coveralls_action(
        self, steps: list[dict]
    ) -> None:
        step = _step_uses(steps, "coverallsapp/github-action@")
        assert step is not None
        assert step["uses"] == "coverallsapp/github-action@v2"

    def test_when_coveralls_step_runs_passes_github_token(
        self, steps: list[dict]
    ) -> None:
        step = _step_uses(steps, "coverallsapp/github-action@")
        assert step is not None
        with_block = step.get("with", {})
        assert "${{ secrets.GITHUB_TOKEN }}" in str(with_block.get("github-token", ""))

    def test_when_coveralls_step_runs_uploads_coverage_xml(
        self, steps: list[dict]
    ) -> None:
        step = _step_uses(steps, "coverallsapp/github-action@")
        assert step is not None
        assert step["with"].get("file") == "coverage.xml"

    def test_when_coveralls_step_fails_continue_on_error_keeps_ci_green(
        self, steps: list[dict]
    ) -> None:
        step = _step_uses(steps, "coverallsapp/github-action@")
        assert step is not None
        assert step.get("continue-on-error") is True


class TestYamlValidity:
    def test_when_workflow_loaded_yaml_is_well_formed(self) -> None:
        loaded = yaml.safe_load(WORKFLOW.read_text())
        assert isinstance(loaded, dict)
        assert "jobs" in loaded
