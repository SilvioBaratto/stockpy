# Cycle 2 — Deterministic Forecasters

## Objective
Implement five deterministic encoder-decoder forecasters — `LSTMForecaster`, `GRUForecaster`, `BiLSTMForecaster`, `BiGRUForecaster`, and `TCNForecaster` — each with separate encoder/decoder sub-modules and teacher-forcing support during training, plus one test file per model verifying instantiation, fit, predict shape, callbacks, and save/load round-trip.

## Project vision
stockpy `0.4.0` becomes a pure time-series forecasting library. Every model accepts a past context window (`context_len`) and predicts a future horizon (`pred_len`), returning arrays of shape `(n_samples, pred_len, n_features)`. No classification, no flat regression, no sklearn mixins. The codebase will be a clean encoder-decoder-only package with deterministic PyTorch models and one probabilistic Pyro model, all sharing the same base class, callback system, and dataset abstraction.

## Preceding cycles
- **Cycle 1** delivered `EncoderDecoderForecaster` (base class with `context_len` and `pred_len`), `TimeSeriesDataset`, package reorganization into `stockpy/forecasters/`, and test scaffolding (`conftest.py`, `test_dataset.py`, `test_callbacks.py`). This cycle must build directly on those artifacts without re-creating them.

## Following cycles
- **Cycle 3** will add `DMMForecaster` (probabilistic) and `TransformerForecaster`, apply full type annotations across all public APIs, and update CI. This cycle should not touch those models or CI.

## In scope
1. `LSTMForecaster` in `forecasters/_lstm.py` — `LSTMEncoder` + `LSTMDecoder`, teacher-forcing during training, autoregressive or one-shot prediction.
2. `GRUForecaster` in `forecasters/_gru.py` — `GRUEncoder` + `GRUDecoder`, same pattern.
3. `BiLSTMForecaster` in `forecasters/_bilstm.py` — bidirectional `LSTMEncoder`, unidirectional `LSTMDecoder`.
4. `BiGRUForecaster` in `forecasters/_bigru.py` — bidirectional `GRUEncoder`, unidirectional `GRUDecoder`.
5. `TCNForecaster` in `forecasters/_tcn.py` — Temporal Convolutional Network encoder (causal dilated convolutions) paired with a `GRU` decoder.
6. Each model must inherit from `EncoderDecoderForecaster`, accept `context_len` and `pred_len` as first-class constructor parameters, and produce `predict()` output of shape `(n_samples, pred_len, n_features)`.
7. One test file per model: `tests/test_lstm_forecaster.py`, `tests/test_gru_forecaster.py`, `tests/test_bilstm_forecaster.py`, `tests/test_bigru_forecaster.py`, `tests/test_tcn_forecaster.py`.
8. Each test file must cover: model instantiation, `fit()` on synthetic sine-wave data for 5 epochs, `predict()` output shape assertion, callback hook firing (EarlyStopping triggers), and model save/load round-trip via safetensors.
9. Update `stockpy/forecasters/__init__.py` to export all five models.
10. Update `stockpy/__init__.py` public API surface to include the new forecasters.

## Out of scope
- `DMMForecaster` and `TransformerForecaster` (Cycle 3).
- Full type annotations (Cycle 3 handles `from __future__ import annotations` and complete hints).
- GitHub Actions CI update (Cycle 3).
- `pyproject.toml` changes beyond adding forecaster modules if needed (Cycle 1 already migrated).
- Changes to `TimeSeriesDataset`, `callbacks/`, `utils/`, `history.py`, or `exceptions.py`.

## Acceptance criteria
- `pytest tests/test_lstm_forecaster.py tests/test_gru_forecaster.py tests/test_bilstm_forecaster.py tests/test_bigru_forecaster.py tests/test_tcn_forecaster.py` passes.
- Each forecaster instantiates with `context_len` and `pred_len`, fits on synthetic data, and `predict()` returns a numpy array of shape `(n_samples, pred_len, n_features)`.
- Each forecaster has separate encoder and decoder sub-modules visible in its architecture.
- Teacher-forcing is implemented during training for LSTM, GRU, BiLSTM, and BiGRU.
- EarlyStopping callback can trigger on at least one forecaster during a short training run.
- Save/load round-trip restores identical weights (verified by a test).
- All five forecasters are importable from `stockpy.forecasters`.