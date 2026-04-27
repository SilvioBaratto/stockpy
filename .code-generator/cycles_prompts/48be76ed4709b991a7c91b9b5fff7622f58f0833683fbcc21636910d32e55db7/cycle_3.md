# Cycle 3 — Advanced Forecasters & QA Polish

## Objective
Implement `DMMForecaster` (probabilistic encoder-decoder refactor of the Deep Markov Model) and `TransformerForecaster` (encoder-decoder Transformer with positional encoding and cross-attention), add full type annotations across all public APIs in `forecasters/`, and update the GitHub Actions CI workflow to run pytest with coverage on Python 3.10–3.12.

## Project vision
stockpy `0.4.0` becomes a pure time-series forecasting library. Every model accepts a past context window (`context_len`) and predicts a future horizon (`pred_len`), returning arrays of shape `(n_samples, pred_len, n_features)`. No classification, no flat regression, no sklearn mixins. The codebase will be a clean encoder-decoder-only package with deterministic PyTorch models and one probabilistic Pyro model, all sharing the same base class, callback system, and dataset abstraction.

## Preceding cycles
- **Cycle 1** delivered the `EncoderDecoderForecaster` base class, `TimeSeriesDataset`, package reorganization, `pyproject.toml`, and test scaffolding.
- **Cycle 2** delivered `LSTMForecaster`, `GRUForecaster`, `BiLSTMForecaster`, `BiGRUForecaster`, and `TCNForecaster` with full test coverage. This cycle must add the remaining two models and finalize code quality.

## Following cycles
None — this is the final cycle. All acceptance criteria from the requirements must be met after this cycle completes.

## In scope
1. `DMMForecaster` in `forecasters/_dmm.py` — reframe the existing Deep Markov Model as a forecasting model: inference network becomes the encoder, generative network becomes the decoder. Remove any remaining classifier variant logic. Keep Pyro SVI training via `TraceMeanField_ELBO`.
2. `TransformerForecaster` in `forecasters/_transformer.py` — encoder-decoder Transformer with positional encoding, multi-head self-attention in encoder, cross-attention in decoder, multi-step ahead forecasting output.
3. Both models must inherit from `EncoderDecoderForecaster`, accept `context_len` and `pred_len`, and return `predict()` shape `(n_samples, pred_len, n_features)`.
4. Add `tests/test_dmm_forecaster.py` and `tests/test_transformer_forecaster.py` covering instantiation, fit, predict shape, callbacks, and save/load round-trip.
5. Add `from __future__ import annotations` and full parameter/return type hints to all public methods in `_base.py` and every forecaster file.
6. Update `.github/workflows/python-package.yml` to run `pytest --cov=stockpy --cov-report=xml` on Python 3.10, 3.11, and 3.12, uploading coverage to Coveralls.
7. Ensure linting passes with Black (88 chars), isort, and ruff.
8. Update `stockpy/forecasters/__init__.py` and `stockpy/__init__.py` to export `DMMForecaster` and `TransformerForecaster`.
9. Verify the full test suite passes: `pytest tests/`.

## Out of scope
- Changes to `TimeSeriesDataset`, `callbacks/`, `utils/`, `history.py`, or `exceptions.py` (stable from Cycle 1).
- Refactoring the five deterministic forecasters from Cycle 2 (unless type-annotation additions require minimal edits).
- Any new models beyond DMM and Transformer.
- Documentation site or notebooks (beyond docstrings and `__init__.py` exports).

## Acceptance criteria
- `pytest tests/` passes entirely.
- `DMMForecaster` and `TransformerForecaster` instantiate, fit, and predict with correct output shape `(n_samples, pred_len, n_features)`.
- `DMMForecaster` uses Pyro SVI and retains probabilistic training; no NNHMM or GHMM remnants exist.
- `TransformerForecaster` includes positional encoding and cross-attention between encoder and decoder.
- All public methods in `_base.py` and every forecaster have complete type annotations.
- CI workflow runs pytest with coverage on Python 3.10, 3.11, and 3.12.
- No sklearn `ClassifierMixin` or `RegressorMixin` remains anywhere in the codebase.
- Version is `0.4.0` in `pyproject.toml`.
- Black (88 chars), isort, and ruff report no issues.