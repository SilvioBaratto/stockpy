# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Direction

stockpy is being restructured (target version `0.4.0`) from a general ML library into a **time-series forecasting library using encoder-decoder architectures only**. The requirements for this restructure are in `.code-generator/requirements.md`. Key constraints:

- **No classification** — all `*Classifier` classes and `ClassifierMixin` are being deleted.
- **No flat regression** — models without temporal structure (MLP, BNN, BCNN) are being deleted.
- **Encoder-decoder only** — every model must accept `context_len` (past) and `pred_len` (future) and produce shape `(n_samples, pred_len, n_features)` from `predict()`.
- **No sklearn dependency** on `ClassifierMixin`/`RegressorMixin` — the base class will be a standalone `EncoderDecoderForecaster` ABC.

## Commands

**Install for development (conda env `stockpy`, Python 3.12):**
```bash
conda activate stockpy
pip install -e .
# or with uv (installed, much faster):
uv pip install -r requirements.txt
```

**Run tests:**
```bash
pytest test/
# single test file:
pytest test/open_ml.py -v
# with coverage:
coverage run -m pytest && coverage report
```

**Lint:**
```bash
pycodestyle stockpy/
black --check stockpy/      # formatter is Black, 88-char lines
```

**Format:**
```bash
black stockpy/
```

**Build docs:**
```bash
cd docs && make html
```

## Architecture

### Current package layout (`stockpy/stockpy/`)

```
base.py                  # Central training engine — Regressor and Classifier base classes
                         # Manages fit/predict loop, DataLoaders, callbacks, SVI (Pyro)
neural_network/          # Pure PyTorch models: MLP, LSTM, BiLSTM, GRU, BiGRU, CNN
                         # Each file has a *Classifier and *Regressor variant
probabilistic/           # Pyro-PPL models: BNN, BCNN, DMM, NNHMM, GHMM
                         # Uses stochastic variational inference (SVI) via guide/model pattern
preprocessing/           # StockDatasetFFNN / RNN / CNN — wraps numpy/pandas into PyTorch Datasets
callbacks/               # Hook system: EarlyStopping, Checkpoint, LRScheduler, EpochScoring, PrintLog
utils/_utils.py          # Device management, tensor↔numpy conversions, parameter filtering
history.py               # History dict-of-lists for per-epoch metric tracking
exceptions.py            # StockpyException hierarchy
```

### How the training loop works (`base.py`)

`base.py` is the core of the library. `Regressor`/`Classifier` inherit from `SkBaseEstimator` and implement `.fit()` / `.predict()`. Key flow:

1. `.fit(X, y)` → `initialize()` → builds model, optimizer, criterion, dataset, callbacks
2. `fit_loop()` → `run_each_epoch()` → `train_step()` / `validation_step()`
3. Probabilistic models use **Pyro SVI** (`TraceMeanField_ELBO`) instead of a standard loss; `base.py` branches on `self._is_probabilistic`.
4. Callbacks are notified at `on_train_begin`, `on_epoch_begin/end`, `on_batch_begin/end`, `on_grad_computed`.
5. Model state is saved/loaded via **safetensors** (not torch.save).

### Callback system

Callbacks in `callbacks/` receive a reference to the `net` (the Regressor/Classifier) and call `net.history` to read metrics. They are plain classes with hook methods — not PyTorch hooks. `_scoring.py` is the most complex: `EpochScoring` re-runs inference on the validation set and stores the metric in `history`.

### Probabilistic models (Pyro)

`DMM`, `NNHMM`, `GHMM` in `probabilistic/` define a Pyro `model` (generative) and `guide` (inference/encoder). The `_combiner.py`, `_emitter.py`, `_transition.py` files are sub-components used by `DMM`. These are the models closest to a true encoder-decoder structure and are the primary candidates for the `DMMForecaster` in the restructure.

### Dataset classes

`preprocessing/_dataset.py` wraps input data into PyTorch `Dataset` objects. `StockDatasetRNN` adds a sliding-window `seq_len` dimension — this is the pattern `TimeSeriesDataset` will replace with `context_len` + `pred_len` dual windows.

## Coding Conventions

- Formatter: **Black** (88 chars). Linter: **pycodestyle**.
- Public model parameters are set in `__init__` and follow the sklearn pattern: stored as-is with the same name as the argument (no mangling), so `get_params()` works automatically.
- Private helpers use a leading underscore prefix (`_fit_loop`, `_train_step`).
- NumPy-style docstrings on all public methods.
- Model sub-modules (encoder, decoder sub-networks) live inside the same `_model.py` file as private classes, not in separate files.
