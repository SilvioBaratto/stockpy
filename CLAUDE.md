# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Direction

stockpy `0.4.0` is a **time-series forecasting library using encoder-decoder architectures only**. The full restructuring spec lives in `.code-generator/requirements.md`. Hard constraints:

- **No classification** — `*Classifier` classes and `ClassifierMixin` removed.
- **No flat regression** — feedforward models without temporal structure (MLP, BNN, BCNN) removed.
- **Encoder-decoder only** — every forecaster accepts `context_len` (past) and `pred_len` (future) and `predict()` returns shape `(n_samples, pred_len, n_features)`.
- **No sklearn `ClassifierMixin`/`RegressorMixin`** — base is the standalone `EncoderDecoderForecaster` ABC.

The README still describes the legacy 0.3.x API (`stockpy.neural_network`, `*Classifier`, `*Regressor`). Treat it as outdated until updated; trust `.code-generator/requirements.md` and the `stockpy/` source.

## Commands

**Install for development (conda env `stockpy`, Python 3.12):**
```bash
conda activate stockpy
pip install -e .
# or with uv (much faster):
uv pip install -r requirements.txt
```

**Tests:**
```bash
pytest test/                                 # full suite (note: `test/`, not `tests/`)
pytest test/test_lstm_forecaster.py -v       # single file
pytest test/test_lstm_forecaster.py::test_fit_predict_shape -v   # single test
coverage run -m pytest && coverage report
```

**Lint / format:**
```bash
pycodestyle stockpy/
black --check stockpy/      # Black, 88-char lines (configured in pyproject.toml)
black stockpy/
```

**Docs:**
```bash
cd docs && make html
```

## Architecture

### Current package layout (`stockpy/`)

```
base.py                  # BaseEstimator (training engine) + EncoderDecoderForecaster ABC
forecasters/             # All concrete models — encoder-decoder only
  _lstm.py _gru.py _bilstm.py _bigru.py _tcn.py _transformer.py
  _dmm.py _dmm_components.py    # DMMForecaster (Pyro SVI) + Combiner/Emitter/Transition
preprocessing/
  _base.py               # ValidSplit, unpack_data, StockpyDataset
  _dataset.py            # TimeSeriesDataset (dual context_len + pred_len sliding windows)
  _transforms.py         # StandardScalerTransform, DifferenceTransform
  _synthetic.py          # synthetic series generators (test fixtures)
callbacks/               # EarlyStopping, Checkpoint, LRScheduler, EpochScoring, PrintLog, EpochTimer
utils/_utils.py          # device mgmt, tensor↔numpy, parameter filtering, check_is_fitted
history.py               # per-epoch metric tracking (dict-of-lists)
exceptions.py            # StockpyException hierarchy
```

The legacy `stockpy/neural_network/` and `stockpy/probabilistic/` packages no longer exist. All models are unified under `forecasters/` and exported from `stockpy.forecasters` via `__init__.py`:
`LSTMForecaster, GRUForecaster, BiLSTMForecaster, BiGRUForecaster, TCNForecaster, DMMForecaster, TransformerForecaster`.

### How the training loop works (`base.py`)

`base.py` is the core. Two classes:

1. **`BaseEstimator`** (line ~239) — sklearn-compatible base. Owns `initialize()`, `fit_loop()`, `run_single_epoch()`, `train_step()`, `validation_step()`, optimizer/criterion plumbing, callbacks, history, safetensors save/load (`save_params` / `load_params`).
2. **`EncoderDecoderForecaster`** (line ~4594) — abstract subclass that adds `context_len` / `pred_len` and the forecasting `fit()` / `predict()` contract. Concrete forecasters in `forecasters/` subclass this.

Flow: `.fit(X, y)` → `initialize()` (builds module, optimizer, criterion, dataset, callbacks) → `fit_loop()` → `run_single_epoch()` → `train_step()` / `validation_step()`.

Probabilistic models (`DMMForecaster`) branch on `self._is_probabilistic` and use **Pyro SVI** with `TraceMeanField_ELBO` instead of a standard PyTorch loss. The Pyro `model` (generative) and `guide` (inference network) double as decoder and encoder.

Model state is persisted via **safetensors**, never `torch.save`.

### Callbacks

Plain Python classes in `callbacks/` with hook methods (`on_train_begin`, `on_epoch_begin/end`, `on_batch_begin/end`, `on_grad_computed`) — not PyTorch hooks. Each receives a reference to the `net` and reads/writes `net.history`. `EpochScoring` in `_scoring.py` is the most complex: re-runs inference on the validation set and stores the metric.

### Datasets

`TimeSeriesDataset` (in `preprocessing/_dataset.py`) replaces the old `StockDatasetRNN/CNN/FFNN`. It produces dual sliding windows of length `context_len` (encoder input) and `pred_len` (decoder target) over a single contiguous series.

## Coding Conventions

- Formatter: **Black** (88 chars). Linter: **pycodestyle**.
- Public model parameters set in `__init__`, stored as-is under the same name (sklearn pattern, so `get_params()` works automatically — no mangling).
- Private helpers use a leading underscore (`_fit_loop`, `_train_step`).
- NumPy-style docstrings on all public methods.
- Encoder/decoder sub-modules live as private classes inside the same `_<model>.py` file as their forecaster, not in separate files. Exception: `DMMForecaster` keeps `_dmm_components.py` because `Combiner`/`Emitter`/`Transition` are reused by Pyro `model`/`guide`.

## Code Generation State

The `.code-generator/` directory contains the active restructure plan (`requirements.md`), generation logs (`logs/`), and run state (`state.json` / `memories/`). Treat it as authoritative for in-flight breaking changes. Don't edit generated state files by hand unless explicitly requested.
