# stockpy

## Description
A Python library for time-series forecasting using encoder-decoder neural architectures. Designed for financial and sequential data, it provides probabilistic and deterministic encoder-decoder models that predict future sequences from past observations — not classifying or regressing on flat features.

## Tech Stack
- **Language**: Python 3.11+
- **Framework**: PyTorch (neural networks), Pyro-PPL (probabilistic encoder-decoders)
- **Database**: none
- **Deploy**: local / pip install
- **Tests**: pytest, pytest-cov

## Features

### Core Restructuring (Breaking Changes)
1. **Remove all classification models** — delete every `*Classifier` class across `neural_network/` and `probabilistic/`. Remove sklearn `ClassifierMixin` inheritance, classification-specific loss functions (`CrossEntropyLoss`, etc.), and classification metrics (`accuracy_score`, `f1_score`) from `base.py`, `callbacks/_scoring.py`, and all model files.
2. **Remove all flat-regression models** — delete `MLPRegressor`, `BNNRegressor`, `BCNNRegressor` and any model whose forward pass does not operate on a temporal sequence. Pure feedforward (MLP, BNN, BCNN) models have no place here.
3. **Rename `Regressor` base to `Forecaster`** — the remaining task is sequence-to-sequence future prediction, not regression on a fixed-size input. Update `base.py`, all model names, `__init__.py` exports, and documentation strings accordingly.

### Encoder-Decoder Architecture (New Core)
4. **Implement a canonical `EncoderDecoderForecaster` base class** in `base.py` (or a new `forecaster.py`) that:
   - Accepts a `context_len` (encoder window, past observations) and `pred_len` (decoder window, future steps to predict).
   - Runs the encoder over the context window, passes the final hidden state to the decoder, and autoregressively (or in one shot) produces `pred_len` future values.
   - Keeps the existing callback system (EarlyStopping, Checkpoint, LRScheduler, PrintLog) fully functional.
5. **Refactor LSTM → `LSTMForecaster`** using the `EncoderDecoderForecaster` base: separate `LSTMEncoder` and `LSTMDecoder` sub-modules, teacher-forcing support during training.
6. **Refactor GRU → `GRUForecaster`** with the same encoder-decoder split and teacher-forcing.
7. **Refactor BiLSTM → `BiLSTMForecaster`** — bidirectional encoder, unidirectional decoder.
8. **Refactor BiGRU → `BiGRUForecaster`** — bidirectional encoder, unidirectional decoder.
9. **Refactor CNN → `TCNForecaster`** — replace the plain CNN with a Temporal Convolutional Network (TCN) encoder (causal dilated convolutions) paired with a GRU decoder.
10. **Refactor DMM → `DMMForecaster`** — keep the Deep Markov Model but reframe it as a pure forecasting model: the inference network is the encoder, the generative network is the decoder. Remove classifier variant, remove NNHMM and GHMM (they add no distinct encoder-decoder value over DMM).
11. **Add `TransformerForecaster`** — encoder-decoder Transformer with positional encoding, multi-head self-attention in the encoder, cross-attention in the decoder, suitable for multi-step ahead forecasting.

### Library Structure
12. **Reorganise the package layout** to match a proper Python library:
    ```
    stockpy/
    ├── __init__.py                  # public API surface
    ├── forecasters/
    │   ├── __init__.py
    │   ├── _base.py                 # EncoderDecoderForecaster ABC
    │   ├── _lstm.py                 # LSTMForecaster
    │   ├── _gru.py                  # GRUForecaster
    │   ├── _bilstm.py               # BiLSTMForecaster
    │   ├── _bigru.py                # BiGRUForecaster
    │   ├── _tcn.py                  # TCNForecaster
    │   ├── _transformer.py          # TransformerForecaster
    │   └── _dmm.py                  # DMMForecaster (probabilistic)
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── _dataset.py              # TimeSeriesDataset (context_len + pred_len windows)
    │   └── _transforms.py           # Normalisation, differencing helpers
    ├── callbacks/                   # keep existing — no changes needed
    ├── utils/
    │   ├── __init__.py
    │   └── _utils.py
    ├── exceptions.py
    └── history.py
    tests/
    ├── conftest.py                  # shared fixtures (synthetic sine-wave data)
    ├── test_lstm_forecaster.py
    ├── test_gru_forecaster.py
    ├── test_bilstm_forecaster.py
    ├── test_bigru_forecaster.py
    ├── test_tcn_forecaster.py
    ├── test_transformer_forecaster.py
    ├── test_dmm_forecaster.py
    ├── test_dataset.py
    └── test_callbacks.py
    ```
13. **`TimeSeriesDataset`** replaces `StockDatasetRNN/CNN/FFNN` with a single class parameterised by `context_len` and `pred_len`; returns `(x_context, y_target)` tensors.
14. **`pyproject.toml`-first packaging** — migrate from `setup.py` to `pyproject.toml` with `[project]` table, proper optional dependency groups (`[dev]`, `[docs]`), and entry-points if needed.

### Testing & Maintainability
15. **pytest suite with fixtures** — `conftest.py` generates a synthetic multivariate sine-wave time series so no network or file I/O is needed in unit tests.
16. **One test file per forecaster** — each file tests: model instantiation, `fit()` on synthetic data (5 epochs), `predict()` output shape `(batch, pred_len, features)`, callback hooks (EarlyStopping triggers), and model save/load round-trip.
17. **`test_dataset.py`** — tests `TimeSeriesDataset` slicing, edge-case sequence lengths, and DataLoader compatibility.
18. **`test_callbacks.py`** — tests EarlyStopping, Checkpoint, LRScheduler, PrintLog in isolation with a mock forecaster.
19. **CI: GitHub Actions** — update `.github/workflows/python-package.yml` to run `pytest --cov=stockpy --cov-report=xml` on Python 3.10, 3.11, 3.12 and upload to Coveralls.
20. **Type annotations** — add `from __future__ import annotations` and full parameter/return type hints to all public methods in `_base.py` and every forecaster.

## Non-functional Requirements
- No sklearn `ClassifierMixin` or `RegressorMixin` anywhere — the library is not a sklearn estimator library.
- All models must support `context_len` and `pred_len` as first-class constructor parameters.
- `predict()` must always return a numpy array of shape `(n_samples, pred_len, n_features)`.
- No test should require internet access or real stock data files.
- Callback system must remain framework-agnostic (works with any `EncoderDecoderForecaster` subclass).
- Code style: Black (88 chars), isort, ruff for linting.

## Project Structure
See feature 12 above for the target folder layout.

## Additional Notes
- **Delete**: `stockpy/neural_network/_mlp.py`, `stockpy/probabilistic/_bnn.py`, `stockpy/probabilistic/_bcnn.py`, `stockpy/probabilistic/_nhmm.py`, `stockpy/probabilistic/_ghmm.py`, all `*Classifier` classes everywhere.
- **Rename**: `stockpy/neural_network/` → `stockpy/forecasters/`; `base.py` `Regressor` → `EncoderDecoderForecaster`.
- **Keep**: `callbacks/`, `utils/`, `exceptions.py`, `history.py` — these are reusable and architecture-agnostic.
- **Pyro** remains a dependency only for `DMMForecaster`; all other models are pure PyTorch.
- The `stock/` sample CSV files (AAPL, TSLA, etc.) can stay as example data for notebooks but must not be required by any test.
- Version bump to `0.4.0` to signal the breaking change.
