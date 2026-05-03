"""Tests for the relocated DMM sub-components.

Coverage for issue #32: ``Combiner``, ``EmitterRegressor`` and ``Transition``
must live under ``stockpy.forecasters._dmm_components`` and the legacy
``stockpy.neural_network`` / ``stockpy.probabilistic`` packages must be gone.
"""

from __future__ import annotations

import importlib

import pytest
import torch


@pytest.fixture
def components():
    return importlib.import_module("stockpy.forecasters._dmm_components")


class TestDmmComponentsModule:
    def test_when_imported_module_exposes_all_three_classes(self, components):
        assert hasattr(components, "Combiner")
        assert hasattr(components, "EmitterRegressor")
        assert hasattr(components, "Transition")

    def test_when_imported_module_declares_all(self, components):
        assert set(components.__all__) == {"Combiner", "EmitterRegressor", "Transition"}


class TestCombinerForward:
    def test_when_called_returns_loc_and_scale_with_expected_shape(self, components):
        combiner = components.Combiner(z_dim=4, rnn_dim=6)
        z_prev = torch.randn(2, 4)
        h_rnn = torch.randn(2, 12)  # rnn_dim * 2
        loc, scale = combiner(z_prev, h_rnn)

        assert loc.shape == (2, 4)
        assert scale.shape == (2, 4)
        assert torch.all(scale > 0)


class TestEmitterRegressorForward:
    def test_when_called_returns_mu_and_sigma_with_expected_shape(self, components):
        emitter = components.EmitterRegressor(
            input_dim=3, z_dim=4, emission_dim=5, output_dim=2
        )
        z_t = torch.randn(7, 4)
        x_t = torch.randn(7, 3)
        mu, sigma = emitter(z_t, x_t)

        assert mu.shape == (7, 2)
        assert sigma.shape == (7, 2)
        assert torch.all(sigma > 0)


class TestTransitionForward:
    def test_when_called_returns_loc_and_scale_with_expected_shape(self, components):
        transition = components.Transition(z_dim=4, input_dim=3, transition_dim=5)
        z_prev = torch.randn(6, 4)
        x_t = torch.randn(6, 3)
        loc, scale = transition(z_prev, x_t)

        assert loc.shape == (6, 4)
        assert scale.shape == (6, 4)
        assert torch.all(scale > 0)


class TestLegacyPackagesRemoved:
    def test_when_importing_legacy_neural_network_raises(self):
        with pytest.raises(ImportError):
            importlib.import_module("stockpy.neural_network")

    def test_when_importing_legacy_probabilistic_raises(self):
        with pytest.raises(ImportError):
            importlib.import_module("stockpy.probabilistic")


class TestDmmForecasterUsesNewComponentsPath:
    def test_when_dmm_module_loaded_components_are_resolved_from_new_path(self):
        dmm = importlib.import_module("stockpy.forecasters._dmm")
        components = importlib.import_module("stockpy.forecasters._dmm_components")

        assert dmm.Combiner is components.Combiner
        assert dmm.EmitterRegressor is components.EmitterRegressor
        assert dmm.Transition is components.Transition
