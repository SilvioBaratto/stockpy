# stockpy 0.4.x — Forecaster Requirements

This document is the authoritative specification for the forecasting models in
`stockpy`. It covers (a) the canonical encoder–decoder forecasters that already
ship in `stockpy/forecasters/`, and (b) two new encoder-only Transformer
forecasters — `PatchTSTForecaster` and `iTransformerForecaster` — to be added.

The architectural rationale: recent literature (Liu et al. 2024, Nie et al.
2023, arXiv 2507.13043) shows that direct-mapping encoder-only Transformers
outperform autoregressive seq2seq on long-horizon multivariate benchmarks
because they avoid error accumulation. Classical seq2seq RNNs and the vanilla
encoder–decoder Transformer remain valuable as pedagogical baselines and on
short horizons. stockpy therefore offers both paradigms behind a single
`EncoderDecoderForecaster` API contract.

---

## 1. Library invariants

All forecasters MUST satisfy the following:

- Subclass `EncoderDecoderForecaster` (`stockpy/base.py:4594`).
- Constructor accepts `context_len: int`, `pred_len: int`, and forwards
  `**kwargs` to `super().__init__`.
- Every public ctor parameter is stored as-is on the instance under the same
  name (sklearn convention — required for `get_params()`).
- `module_.forward(x, y=None) -> Tensor` returns shape
  `(batch, pred_len, n_features)`.
- `predict(X)` returns `numpy.ndarray` of shape
  `(n_windows, pred_len, n_features)`.
- Model state persisted via **safetensors** only (`save_params` /
  `load_params`). Never `torch.save`.
- Probabilistic models set `self._is_probabilistic = True` and use Pyro SVI;
  the new models specified in §3–§4 are deterministic.
- File naming: `stockpy/forecasters/_<name>.py`, class `<Name>Forecaster`.
  Encoder/decoder/backbone sub-modules live as private classes inside the same
  file (exception: `_dmm_components.py`, kept separate because `Combiner` /
  `Emitter` / `Transition` are reused by Pyro `model` and `guide`).
- First import in every `_*.py` is `from __future__ import annotations`.
- NumPy-style docstrings on all public methods.
- Black 88-char line length; pycodestyle clean.
- Full type annotations on `__init__`, `forward`, and `initialize_module`.
  `initialize_module` returns `Self` (or the concrete subclass).
- `model_type` attribute returns `"rnn"` (drives the data-iterator selector in
  the base class; the name is historical and applies to all sequence
  forecasters regardless of internal architecture). See
  `stockpy/forecasters/_transformer.py:304-309` for the canonical comment.

### 1.1 Required base-class hooks (every concrete forecaster)

The following overrides are non-optional. They are present in every existing
forecaster and must be replicated identically in the new ones.

- `self._modules = ["module"]` set in `__init__` so
  `_initialize_module` (`base.py:1271-1298`) can locate `module_` for
  device placement.
- Override `state_dict()` and `load_state_dict()` to delegate to
  `self.module_` (see `_lstm.py:240-248`, `_transformer.py:296-300`).
  Without this, safetensors save/load silently round-trips the wrong tensors.
- Override `_set_training(training)` to call `self.module_.train(training)`
  (see `_lstm.py:206-208`). Without this, dropout stays active during
  `predict()`.
- Define `__sklearn_tags__` and `_get_tags` identically to the existing
  `TransformerForecaster` (`_transformer.py:247-257`).

### 1.2 Constructor invariants for new (non-teacher-forcing) forecasters

Models in §3 and §4 are direct-multi-step. They do **not** override
`train_step_single`. The base `train_step_single` (`base.py:1769-1773`) does
not pass `y` to `forward`, so the "forward ignores `y`" contract is satisfied
without further wiring. Implementors MUST NOT add a teacher-forcing override
for these models.

---

## 2. Canonical encoder–decoder forecasters (existing)

These ship today in `stockpy/forecasters/`. They are the reference
implementation of the **encoder–decoder seq2seq** paradigm. No code changes
required by this document — they are listed here so the spec is complete.

| Forecaster              | File              | Encoder                                    | Decoder                                  | Decode strategy                                      |
| ----------------------- | ----------------- | ------------------------------------------ | ---------------------------------------- | ---------------------------------------------------- |
| `LSTMForecaster`        | `_lstm.py`        | `nn.LSTM`                                  | `nn.LSTM` + linear output projection     | Teacher forcing during training, autoregressive at inference |
| `GRUForecaster`         | `_gru.py`         | `nn.GRU`                                   | `nn.GRU` + linear output projection      | Same as LSTM                                         |
| `BiLSTMForecaster`      | `_bilstm.py`      | bidirectional `nn.LSTM`                    | unidirectional `nn.LSTM`                 | Hidden state projected before decoder                |
| `BiGRUForecaster`       | `_bigru.py`       | bidirectional `nn.GRU`                     | unidirectional `nn.GRU`                  | Hidden state projected before decoder                |
| `TransformerForecaster` | `_transformer.py` | `nn.TransformerEncoder` + sinusoidal posenc | `nn.TransformerDecoder` (cross-attn)     | Causal `tgt_mask`; teacher-forced training, autoregressive inference |
| `TCNForecaster`         | `_tcn.py`         | dilated causal `Conv1d` stack              | `nn.GRU` decoder                         | Encoder produces context vector; GRU decodes         |
| `DMMForecaster`         | `_dmm.py`         | RNN encoder (`guide`)                      | Markov latent emission (`model`)         | Pyro SVI with `TraceMeanField_ELBO`                  |

Constructor signatures, default hyperparameters, and decode logic are defined
in those files and are considered frozen for this release.

---

## 3. New forecaster: `PatchTSTForecaster`

**Reference**: Nie, Nguyen, Sinthong, Kalagnanam,
*A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*,
ICLR 2023.

**Paradigm**: encoder-only Transformer with **channel-independent**
processing and a direct multi-step head. The encoder–decoder ABC is satisfied
by the `context_len → pred_len` mapping contract; there is no literal decoder
module.

### 3.1 File and class

- New file: `stockpy/forecasters/_patchtst.py`.
- Class: `PatchTSTForecaster(EncoderDecoderForecaster)`.

### 3.2 Constructor

```python
def __init__(
    self,
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 128,
    nhead: int = 16,
    num_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: float = 0.2,
    activation: str = "gelu",
    revin: bool = True,
    revin_affine: bool = True,
    context_len: int = 96,
    pred_len: int = 24,
    **kwargs,
) -> None: ...
```

All parameters stored as-is on `self`.

**`__init__` validation (raise `ValueError` on bad combinations):**

- `context_len < patch_len` → unfold impossible.
- `d_model % nhead != 0` → `nn.MultiheadAttention` requires divisibility.
- `patch_len < 1` or `stride < 1` → invalid window.

**Default note:** `dim_feedforward = 4 * d_model` follows standard Transformer
ratios (paper uses GELU, ratio ≈ 4×). `nhead=16, d_model=128` gives head_dim=8
matching Nie et al. ICLR 2023.

### 3.3 Architecture (forward pass)

Input shape `(B, L, C)` where `L == context_len`, `C == n_features`.

1. **RevIN** (reversible instance normalization), if `revin=True`. Compute
   per-(batch, feature) mean and standard deviation over the time axis;
   subtract and divide. Return `(x_norm, mean, std)` from the RevIN
   `_normalize` call; pass `mean` and `std` as **local tensors through the
   call stack** to the inverse step. Do NOT cache statistics as `nn.Module`
   attributes — that breaks under repeated/concurrent forward passes.
2. Permute to `(B, C, L)`.
3. **Patching with end-padding**: replicate-pad the right edge of the time
   axis so that `(L_padded - patch_len)` is divisible by `stride`, matching
   the original PatchTST reference implementation. Then unfold with window
   `patch_len` and `stride`. Result shape `(B, C, N, patch_len)` where
   `N = floor((L_padded - patch_len) / stride) + 1`. With default
   `context_len=96, patch_len=16, stride=8`, `L_padded=104` and `N=12`.
4. **Channel independence**: reshape to `(B * C, N, patch_len)`. The
   Transformer backbone weights are shared across all `C` features.
5. **Patch embedding**: `nn.Linear(patch_len, d_model)` + learnable positional
   embedding of shape `(N, d_model)`.
6. **Transformer encoder**: `nn.TransformerEncoder` with `num_layers` layers,
   post-norm, `batch_first=True`. Activation `gelu` by default.
7. **Flatten head**: flatten `(B * C, N * d_model)` → `nn.Linear(N * d_model,
   pred_len)` → `(B * C, pred_len)`.
8. Reshape to `(B, C, pred_len)`. Permute to `(B, pred_len, C)`. If RevIN
   active, multiply by `std` and add `mean` (the local tensors threaded
   through from step 1).

### 3.4 Private modules in `_patchtst.py`

All sub-modules are private (leading underscore) per §1. Only
`PatchTSTForecaster` appears in this module's `__all__`.

```python
class _RevIN(nn.Module):
    """Reversible instance norm. Affine gamma/beta lazily allocated on
    first forward call once n_features is known. eps=1e-5."""
    def __init__(self, num_features: int, affine: bool = True,
                 eps: float = 1e-5) -> None: ...
    def _normalize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...
    def _denormalize(self, x: Tensor, mean: Tensor, std: Tensor) -> Tensor: ...

class _PatchEmbedding(nn.Module):
    """Unfold + linear + learnable positional embedding."""
    def __init__(self, patch_len: int, stride: int, d_model: int,
                 num_patches: int) -> None: ...

class _PatchTSTModel(nn.Module):
    """Top-level module used by the forecaster. Composes RevIN +
    patch embedding + nn.TransformerEncoder + flatten head."""
    def __init__(self, n_features: int, context_len: int, pred_len: int,
                 patch_len: int, stride: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int, dropout: float,
                 activation: str, revin: bool,
                 revin_affine: bool) -> None: ...
    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor: ...
```

`gamma` and `beta` of `_RevIN` are learnable `nn.Parameter` tensors of shape
`(num_features,)` when `affine=True`; allocated in `_PatchTSTModel.__init__`
because `n_features` is known there (passed from `initialize_module` via
`self.n_features_in_`).

### 3.5 Training and loss

- `initialize_module` builds `_PatchTSTModel` with stored hyperparameters
  plus `self.n_features_in_` and `self.context_len`, then sets
  `self.criterion_ = nn.MSELoss()`.
- `forward` ignores `y` — direct multi-step head, no teacher forcing,
  no `train_step_single` override (see §1.2).
- **Recommended optimizer: Adam.** SGD (the base default) will not converge
  for this architecture. Spec consumers SHOULD pass
  `optimizer=torch.optim.Adam` and `lr=1e-3` at fit time, or override
  defaults at the forecaster ctor. The `test_fit_on_synthetic_data` test
  MUST pass `optimizer=Adam`.

---

## 4. New forecaster: `iTransformerForecaster`

**Reference**: Liu, Hu, Liu, Zhang, Wang, Long,
*iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting*, ICLR 2024.

**Paradigm**: encoder-only Transformer where each **variate** (feature) is one
token. Self-attention is computed across variates, capturing
multivariate cross-feature dependence. There is no positional encoding because
variates are an unordered set.

### 4.1 File and class

- New file: `stockpy/forecasters/_itransformer.py`.
- Class: `iTransformerForecaster(EncoderDecoderForecaster)`.

### 4.2 Constructor

```python
def __init__(
    self,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    activation: str = "gelu",
    use_norm: bool = True,
    context_len: int = 96,
    pred_len: int = 24,
    **kwargs,
) -> None: ...
```

**`__init__` validation (raise `ValueError` on bad combinations):**

- `d_model % nhead != 0` → `nn.MultiheadAttention` requires divisibility.
- `context_len < 1` or `pred_len < 1` → invalid window.

### 4.3 Architecture (forward pass)

Input shape `(B, L, C)`.

1. **Series-wise normalization**, if `use_norm=True`. Compute per-(batch,
   feature) mean and standard deviation over the time axis; subtract and
   divide. Pass `mean` and `std` as **local tensors through the call stack**
   to step 6. Do NOT store on `nn.Module`.
2. Permute to `(B, C, L)`.
3. **Variate embedding**: `nn.Linear(L, d_model)` applied per variate →
   `(B, C, d_model)`. Each of the `C` rows is now one token.
4. **Transformer encoder**: `nn.TransformerEncoder` with `num_layers` layers,
   `batch_first=True`. **No positional encoding** — variates are unordered.
   Self-attention thus attends across variates rather than across time.
5. **Projection head**: `nn.Linear(d_model, pred_len)` per token →
   `(B, C, pred_len)`.
6. Permute to `(B, pred_len, C)`. If normalization active, restore using
   `mean` and `std` (local tensors threaded through from step 1).

### 4.4 Private modules in `_itransformer.py`

All sub-modules are private (leading underscore) per §1. Only
`iTransformerForecaster` appears in this module's `__all__`.

```python
class _iTransformerEmbedding(nn.Module):
    """Per-variate linear projection L -> d_model. Shared weights across
    all C variates."""
    def __init__(self, context_len: int, d_model: int) -> None: ...

class _iTransformerModel(nn.Module):
    """Top-level module. Composes per-variate embedding +
    nn.TransformerEncoder (no posenc) + per-variate projection head."""
    def __init__(self, context_len: int, pred_len: int, d_model: int,
                 nhead: int, num_layers: int, dim_feedforward: int,
                 dropout: float, activation: str, use_norm: bool) -> None: ...
    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor: ...
```

The projection head is a single shared `nn.Linear(d_model, pred_len)` applied
per token; weights are not per-variate. `n_features_in_` is therefore not
needed at module construction time.

### 4.5 Training and loss

- `initialize_module` builds `_iTransformerModel` with stored hyperparameters
  plus `self.context_len`, then sets `self.criterion_ = nn.MSELoss()`.
- `forward` ignores `y`. No `train_step_single` override (see §1.2).
- **Recommended optimizer: Adam.** Same rationale as PatchTST.
  `test_fit_on_synthetic_data` MUST pass `optimizer=Adam`.

---

## 5. Public API registration

Update `stockpy/forecasters/__init__.py`:

- Add imports:
  ```python
  from ._patchtst import *
  from ._itransformer import *
  ```
- Append to `__all__`:
  ```python
  "PatchTSTForecaster",
  "iTransformerForecaster",
  ```

Each new `_*.py` file declares its own `__all__` listing only the
`<Name>Forecaster` symbol.

No changes to `pyproject.toml` (no new runtime dependencies — both models use
`torch.nn.TransformerEncoder` and standard ops already pulled in).

---

## 6. Tests

For each new forecaster, create
`test/test_<name>_forecaster.py` mirroring
`test/test_transformer_forecaster.py`. Required cases:

1. `test_is_subclass_of_encoder_decoder_forecaster` — `issubclass` check.
2. `test_instantiation_sets_context_and_pred_len` — ctor wiring.
3. `test_model_type_attribute` — equals `"rnn"`.
4. `test_initialize_module_creates_module_` — `module_` and `criterion_` set.
5. Architecture-specific:
   - **PatchTST**:
     - `test_patch_count_correct` — with end-padding,
       N = floor((L_padded − patch_len)/stride) + 1; assert against the
       backbone's stored `num_patches` attribute.
     - `test_revin_round_trip` — fit a model on a constant series and verify
       output equals input within `atol=1e-5`.
     - `test_channel_independence_shared_weights` — build two models with
       `n_features=1` and `n_features=8`, assert
       `sum(p.numel() for p in model.module_.parameters())` differs only by
       the RevIN affine params (`2 * n_features` when `revin_affine=True`).
     - `test_invalid_ctor_raises` — `context_len < patch_len`,
       `d_model % nhead != 0`, `patch_len < 1`, `stride < 1` each raise
       `ValueError`.
   - **iTransformer**:
     - `test_variate_token_axis` — register a forward hook on the inner
       `TransformerEncoder` and assert its input has shape `(B, C, d_model)`,
       not `(B, L, d_model)`.
     - `test_invalid_ctor_raises` — `d_model % nhead != 0` raises `ValueError`.
6. `test_forward_output_shape` — output is `(B, pred_len, n_features)`.
7. `test_predict_returns_correct_shape` — predict on synthetic series.
8. `test_fit_on_synthetic_data` — uses `stockpy.preprocessing._synthetic`,
   passes `optimizer=torch.optim.Adam`, `lr=1e-3`; loss decreases monotonically
   over the first 5 epochs (use a small fixed seed).
9. `test_save_load_roundtrip(tmp_path)` — instantiate with `dropout=0.0`,
   call `model.module_.eval()` before both prediction calls, then assert
   predictions match within `atol=1e-6`. (Stochastic dropout would otherwise
   make the round-trip flaky.)

The annotation contract test (`test/test_forecaster_annotations.py`) discovers
new files in `stockpy/forecasters/` automatically; both new files MUST
therefore start with `from __future__ import annotations` and provide complete
type hints on `__init__`, `forward`, and `initialize_module`.

---

## 7. README update

Append two rows to the model table at `README.md:26-36`:

| Model                    | Architecture                                                                |
| ------------------------ | --------------------------------------------------------------------------- |
| `PatchTSTForecaster`     | Encoder-only Transformer over patched series, channel-independent. Direct multi-step head. |
| `iTransformerForecaster` | Encoder-only Transformer with variates as tokens. Per-variate direct projection to `pred_len`. |

---

## 8. Acceptance criteria

- `pytest test/` passes, including the two new test files, with no
  regressions in existing forecaster tests.
- `from stockpy.forecasters import PatchTSTForecaster, iTransformerForecaster`
  succeeds.
- `model.fit(X, y).predict(X)` returns shape
  `(n_windows, pred_len, n_features)` for both new models.
- `model.save_params(path); model.load_params(path)` round-trips weights via
  safetensors; reloaded model produces predictions equal to the original
  within `atol=1e-6`.
- `pycodestyle stockpy/` clean.
- `black --check stockpy/` clean.
- Annotation coverage test passes for both new files.

---

## 9. Out of scope

- Channel-mixing PatchTST variant (paper reports channel-independent as the
  stronger baseline).
- Time-series foundation models (TimesFM, Chronos, MOIRAI-2) — separate spec.
- Probabilistic variants of either new model — DMMForecaster covers the Pyro
  SVI track.
- Refactor of existing encoder–decoder forecasters listed in §2.
