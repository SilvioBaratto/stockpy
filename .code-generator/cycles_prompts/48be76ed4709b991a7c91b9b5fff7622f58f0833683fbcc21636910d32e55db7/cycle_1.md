# Cycle 1 — Foundation & Core Infrastructure

## Objective
Remove every classifier and flat-regression model from the codebase, introduce the `EncoderDecoderForecaster` abstract base class, reorganize the package into `stockpy/forecasters/`, migrate packaging to `pyproject.toml`, and create the shared `TimeSeriesDataset` plus test scaffolding (`conftest.py`, `test_dataset.py`, `test_callbacks.py`) so later cycles have a solid foundation to build on.

## Project vision
stockpy `0.4.0` becomes a pure time-series forecasting library. Every model accepts a past context window (`context_len`) and predicts a future horizon (`pred_len`), returning arrays of shape `(n_samples, pred_len, n_features)`. No classification, no flat regression, no sklearn mixins. The codebase will be a clean encoder-decoder-only package with deterministic PyTorch models and one probabilistic Pyro model, all sharing the same base class, callback system, and dataset abstraction.

## Preceding cycles
None — this is the first cycle. It must establish everything later cycles depend on.

## Following cycles
- **Cycle 2** depends on this cycle for the `EncoderDecoderForecaster` base, `TimeSeriesDataset`, and test fixtures.
- **Cycle 3** depends on this cycle for the package layout, CI skeleton, and typing conventions established here.

## In scope
1. Delete all `*Classifier` classes and `ClassifierMixin` usage across `neural_network/` and `probabilistic/`.
2. Delete flat-regression models: `MLPRegressor`, `BNNRegressor`, `BCNNRegressor` and their files (`_mlp.py`, `_bnn.py`, `_bcnn.py`).
3. Delete `NNHMM` and `GHMM` from `probabilistic/`.
4. Rename `Regressor` → `EncoderDecoderForecaster` in `base.py` (or a new `forecaster.py` inside `forecasters/`). The base must define `context_len` and `pred_len` as constructor parameters and enforce `predict()` shape `(n_samples, pred_len, n_features)`.
5. Reorganize package layout: create `stockpy/forecasters/`, move remaining model logic there as stub files (`_lstm.py`, `_gru.py`, etc.), update `stockpy/__init__.py`.
6. Migrate from `setup.py` to `pyproject.toml` with `[project]` table, optional `[dev]` and `[docs]` groups, and version `0.4.0`.
7. Implement `TimeSeriesDataset` in `preprocessing/_dataset.py` parameterized by `context_len` and `pred_len`, returning `(x_context, y_target)` tensors. Remove old `StockDatasetFFNN`, `StockDatasetRNN`, `StockDatasetCNN`.
8. Add `preprocessing/_transforms.py` with normalization / differencing helpers.
9. Create `tests/conftest.py` with a synthetic multivariate sine-wave fixture (no network I/O).
10. Create `tests/test_dataset.py` covering slicing, edge-case lengths, and DataLoader compatibility.
11. Create `tests/test_callbacks.py` covering EarlyStopping, Checkpoint, LRScheduler, PrintLog with a mock forecaster.
12. Keep `callbacks/`, `utils/`, `exceptions.py`, `history.py` intact — they are reusable.

## Out of scope
- Implementing any concrete forecaster model logic (LSTM, GRU, BiLSTM, BiGRU, TCN, Transformer, DMM). Stub files only.
- Type annotations across public APIs (Cycle 3 handles full typing).
- GitHub Actions CI workflow update (Cycle 3).
- DMM probabilistic refactoring beyond deletion of NNHMM/GHMM.
- Any notebook or documentation rewrite beyond docstring updates in `base.py`.

## Acceptance criteria
- `pytest tests/test_dataset.py` and `pytest tests/test_callbacks.py` pass.
- No `*Classifier` class or `ClassifierMixin` remains in the codebase.
- No `MLPRegressor`, `BNNRegressor`, or `BCNNRegressor` remains.
- `EncoderDecoderForecaster` exists and defines `context_len`, `pred_len`, and an abstract `predict()` contract returning shape `(n_samples, pred_len, n_features)`.
- `TimeSeriesDataset` correctly yields context/target windows of the requested lengths.
- `pyproject.toml` is present and `pip install -e .` works in a fresh environment.
- Package imports cleanly: `import stockpy` and `from stockpy.forecasters import EncoderDecoderForecaster` succeed.