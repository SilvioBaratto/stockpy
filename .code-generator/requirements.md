# stockpy 0.4.0 — Identified Bugs

Bugs surfaced while building `examples/quickstart.ipynb` against the bundled `stock/AAPL.csv` (1569 OHLCV rows, `LSTMForecaster(context_len=30, pred_len=5)`).

---

## Bug 1 — `safetensors` load fails because `self.device` is a `torch.device` object

**Severity:** High (blocks the documented save/load path)

**Location:** `stockpy/base.py:3284`

```python
with safe_open(f_name, framework="pt", device=self.device) as f:
```

**Root cause:** `self.device` is set in `BaseEstimator.__init__` (`base.py:297`) as `torch.device("cuda" if ... else "cpu")` — i.e. a `torch.device` object. `safetensors.safe_open` accepts a `str` (`"cpu"`, `"cuda:0"`, ...) or an `int`, not a `torch.device`. The repr `device(type='cpu')` does not match safetensors' device parser.

**Repro:**

```python
m = LSTMForecaster(context_len=30, pred_len=5)
m.fit(X, y=X, epochs=0, train_split=None, verbose=0)
m.save_params(f_params="ckpt.safetensors", use_safetensors=True)

m2 = LSTMForecaster(context_len=30, pred_len=5)
m2.fit(X, y=X, epochs=0, train_split=None, verbose=0)
m2.load_params(f_params="ckpt.safetensors", use_safetensors=True)
```

**Observed:**

```
SafetensorError: device cpu is invalid
```

**Workaround in user code:** `m2.device = str(m2.device)` before `load_params`.

**Suggested fix:** in `_get_state_dict` at `base.py:3284` (and the analogous save-path device handling), coerce with `str(self.device)` or `self.device.type if isinstance(self.device, torch.device) else self.device`.

**Test gap:** `test/test_lstm_forecaster.py::test_save_load_roundtrip` uses `save_params`/`load_params` with the **default** torch save (no `use_safetensors=True`), so the bug is not exercised. Add a `use_safetensors=True` variant to that test to catch regressions.

---

## Bug 2 — Default `train_split=ValidSplit(5)` is incompatible with windowed datasets

**Severity:** High (blocks the canonical `fit(X, y=X)` invocation)

**Locations:**
- Default set at `stockpy/base.py:4691` — `train_split=ValidSplit(5)` in `EncoderDecoderForecaster.fit`.
- Length check at `stockpy/preprocessing/_base.py:637`.

**Root cause:** `TimeSeriesDataset` derives windows of length `(series_len - context_len - pred_len) // stride + 1`, which is strictly less than `len(y)` when the user passes `y=X` (the natural auto-regressive call). `ValidSplit.__call__` then compares `get_len(dataset) != get_len(y)` and raises:

```
ValueError: Cannot perform a CV split if dataset and y have different lengths.
```

**Repro:**

```python
m = LSTMForecaster(context_len=30, pred_len=5)
m.fit(X_train, y=X_train, epochs=15, batch_size=32, lr=1e-3,
      optimizer=torch.optim.Adam)   # default train_split fires the error
```

**Workaround in user code:** pass `train_split=None` (used in the example notebook and in `test/test_lstm_forecaster.py`).

**Suggested fix:** `EncoderDecoderForecaster.fit` should either
- compare lengths against the **windowed dataset** rather than the raw `y`, or
- detect the auto-regressive case (`y is X` / `y is None`) and skip the length comparison, or
- ship `train_split=None` as the default and document `ValidSplit` as opt-in.

**Related memory:** observation #842 already noted the small-dataset failure mode of `ValidSplit(5)`.

---

## Bug 3 — `StandardScalerTransform` silently changes the array library (numpy → torch)

**Severity:** Medium (UX / type-stability surprise)

**Locations:** `stockpy/preprocessing/_transforms.py:103, 126, 142`

```python
return torch.from_numpy(scaled).float()      # transform
return torch.from_numpy(original).float()    # inverse_transform
return self.fit(data).transform(data)        # fit_transform inherits the cast
```

**Root cause:** All three methods accept `array-like` (numpy/list/DataFrame) but always return `torch.Tensor`. Downstream numpy-only code (`array.astype(...)`, `np.reshape`, indexing helpers) fails with `AttributeError: 'Tensor' object has no attribute 'astype'`.

**Repro:**

```python
scaler = StandardScalerTransform()
X_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
# AttributeError: 'Tensor' object has no attribute 'astype'
```

**Workaround in user code:** explicit `.numpy()` after every call.

**Suggested fix:** return the same array library as the input (mirror sklearn's `StandardScaler`). Convert to tensor only at the model boundary in `to_tensor` (`utils/_utils.py`).

---

## Bug 4 — `TimeSeriesDataset` raises on series shorter than `context_len + pred_len`, but the error fires deep inside `fit`

**Severity:** Low (correct behavior, poor diagnostics)

**Location:** `stockpy/preprocessing/_dataset.py:127`

**Root cause:** The check `series_len < min_required` is enforced inside `__init__`, which is invoked only after `fit` builds the dataset. Users with short test sets get a confusing traceback originating in the training loop instead of an upfront validation error.

**Suggested fix:** validate `len(X) >= context_len + pred_len` in `EncoderDecoderForecaster.fit` (or `check_data`) before any dataset/iterator construction, with a message that names both `context_len` and `pred_len`.

**Related memory:** observation #843 (sliding windows produce few samples from short series).

---

## Bug 5 — Legacy packaging artifacts after the 0.3.x → 0.4.0 restructure

**Severity:** Low (housekeeping)

**Symptoms:** `setup.py` is missing / out of sync (memory #839); `stockpy_learn.egg-info/` and `build/` are checked-in stale; the README's old code samples still reference `stockpy.neural_network` / `stockpy.probabilistic` until the recent rewrite.

**Suggested fix:** confirm `pyproject.toml` is the single source of truth (it is), `git rm -r build/ stockpy_learn.egg-info/`, add both to `.gitignore`, and verify `python -m build` still produces a clean wheel.

---

## Verification checklist for fixes

- `pytest test/ -x` green.
- `examples/quickstart.ipynb` runs end-to-end via `jupyter nbconvert --execute` with **no** workarounds (the `reloaded.device = str(...)` line in cell 15 should become unnecessary after Bug 1 is fixed).
- New regression tests:
  - `test/test_save_load.py::test_safetensors_roundtrip_cpu` — covers Bug 1.
  - `test/test_valid_split.py::test_default_train_split_with_autoregressive_y` — covers Bug 2.
  - `test/test_transforms.py::test_standard_scaler_returns_numpy_for_numpy_input` — covers Bug 3.
