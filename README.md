<div align="center">
  <a href="https://github.com/SilvioBaratto/stockpy"> <img width=600 src="docs/source/_static/img/stockpi_v3.svg"></a>
</div>

![Python package](https://github.com/SilvioBaratto/stockpy/workflows/Python%20package/badge.svg?branch=master)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<img src='https://img.shields.io/badge/Code%20style-Black-%23000000'/>
[![Documentation Status](https://readthedocs.org/projects/stockpy/badge/?version=latest)](https://stockpy.readthedocs.io/?badge=latest)
[![PyPI version](https://badge.fury.io/py/stockpy-learn.svg)](https://badge.fury.io/py/stockpy-learn)

## Table of Contents
* [Description](#description)
* [Documentation](https://stockpy.readthedocs.io/)
* [Installation](#dependencies-and-installation)
* [Usage](#usage)
* [Data Downloader](#data-downloader)
* [License](#license)
* [Contributing](#how-to-contribute)
* [TODOs](#todos)

## Description
**stockpy** (`0.4.0`) is a Python library for **time-series forecasting using encoder-decoder neural architectures**. Each model consumes a context window of past observations of length `context_len` and produces a prediction window of `pred_len` future steps with shape `(n_samples, pred_len, n_features)`. The library is built on PyTorch, with Pyro-PPL powering the probabilistic forecasters via stochastic variational inference.

> **Breaking changes vs. 0.3.x**: classification (`*Classifier`), flat regression (`MLPRegressor`, `BNNRegressor`, `BCNNRegressor`), and the `stockpy.neural_network` / `stockpy.probabilistic` namespaces have all been removed. All models now live under `stockpy.forecasters` and inherit from `EncoderDecoderForecaster`.

Available forecasters:

| Model                  | Architecture                                                       |
|------------------------|--------------------------------------------------------------------|
| `LSTMForecaster`       | LSTM encoder + LSTM decoder with teacher forcing                   |
| `GRUForecaster`        | GRU encoder + GRU decoder with teacher forcing                     |
| `BiLSTMForecaster`     | Bidirectional LSTM encoder + unidirectional LSTM decoder           |
| `BiGRUForecaster`      | Bidirectional GRU encoder + unidirectional GRU decoder             |
| `TCNForecaster`        | Temporal Convolutional Network (causal dilated) encoder + GRU dec. |
| `TransformerForecaster`| Encoder-decoder Transformer with multi-head self/cross attention   |
| `DMMForecaster`        | Deep Markov Model (Pyro SVI) — probabilistic encoder-decoder       |

## Usage
Import the forecaster you want from `stockpy.forecasters`. Inputs must be 2-D arrays shaped `(time_steps, n_features)`; the library handles the sliding-window split into `context_len` / `pred_len` pairs internally via `TimeSeriesDataset`.

### Deterministic forecasting (LSTM)

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from stockpy.forecasters import LSTMForecaster

# Load the dataset
df = pd.read_csv("stock/AAPL.csv", parse_dates=True, index_col="Date").dropna(how="any")
features = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)

# Chronological split
n_train = int(len(features) * 0.8)
X_train, X_test = features[:n_train], features[n_train:]

# Scale on the training window
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Fit an encoder-decoder LSTM: 30 past steps -> 5 future steps
model = LSTMForecaster(
    context_len=30,
    pred_len=5,
    rnn_size=64,
    hidden_size=64,
    num_layers=2,
    dropout=0.1,
)
model.fit(
    X_train,
    y=None,                        # auto-regressive: target derived from X
    epochs=50,
    batch_size=32,
    lr=1e-3,
    optimizer=torch.optim.Adam,
)

# predict() returns shape (n_windows, pred_len, n_features)
forecasts = model.predict(X_test)
print(forecasts.shape)
```

### Probabilistic forecasting (DMM)

```python
from stockpy.forecasters import DMMForecaster

model = DMMForecaster(
    context_len=30,
    pred_len=5,
    z_dim=16,
    emission_dim=32,
    transition_dim=32,
    rnn_dim=32,
)
model.fit(X_train, epochs=30, batch_size=32, lr=1e-3, optimizer=torch.optim.Adam)
samples = model.predict(X_test)   # (n_windows, pred_len, n_features)
```

The same pattern applies to every forecaster — change the import and the model-specific hyperparameters; `context_len` / `pred_len` / `fit` / `predict` are uniform across the API.

### Saving and loading

Model weights are persisted with **safetensors**, not `torch.save`:

```python
model.save_params(f_params="lstm.safetensors")
loaded = LSTMForecaster(context_len=30, pred_len=5).initialize()
loaded.load_params(f_params="lstm.safetensors")
```

## Dependencies and installation
**stockpy** requires Python ≥ 3.10 and the packages listed in `pyproject.toml` (PyTorch, Pyro-PPL, NumPy, pandas, scikit-learn, safetensors, tqdm, matplotlib, tabulate, yfinance). It can be installed via `pip` or directly from source.

### Installing via pip

```bash
pip install stockpy-learn
```

To uninstall:

```bash
pip uninstall stockpy-learn
```

### Installing from source

```bash
git clone https://github.com/SilvioBaratto/stockpy
cd stockpy
pip install -e .
# or, for a much faster install via uv:
uv pip install -r requirements.txt
```

### Development install

```bash
pip install -e ".[dev]"      # adds pytest, pytest-cov, black, ruff, pycodestyle
pytest test/                 # run the test suite
black stockpy/               # format
pycodestyle stockpy/         # lint
```

## Data downloader
`data_downloader.py` is a command-line utility for fetching and updating per-ticker OHLCV CSVs directly from Yahoo Finance via the [`yfinance`](https://github.com/ranaroussi/yfinance) library. Output schema is auto-adjusted (`Open, High, Low, Close, Volume`, splits/dividends already applied via `auto_adjust=True`). Tested on Ubuntu 22.04 LTS.

### Behaviour highlights

- **Symbol normalization** — Yahoo Finance uses `-` in tickers, the Wikipedia S&P 500 table uses `.`. Inputs like `BRK.B` are auto-converted to `BRK-B` and saved as `BRK-B.csv`. Whitespace and `=` are stripped.
- **Lazy S&P universe fetch** — the Wikipedia constituent list is only downloaded when actually needed (`--download`, `--range`, or empty target folder). Single-stock and update flows skip the scrape.
- **Incremental updates** — `--update` / `--update-stock` resume from the last date in the existing CSV (`last_index + 1 day`) and append, dedup, and sort. No-ops cleanly when already up-to-date.
- **Batch downloads** — `--download` and `--stock-list` use a single `yf.download(..., group_by="ticker")` call with threading, then split per symbol.
- **Persistence** — CSVs use `Date` as the index column, written via `os.path.join` so paths work on any OS. Folder is created with `os.makedirs(..., exist_ok=True)`.

### CLI flags

| Parameter         | Explanation                                                                                                                                                                                                                  |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--download`      | Download every S&P 500 stock as separate CSVs. Default date range: `2017-01-01` to today.                                                                                                                                    |
| `--stock`         | Download one specific ticker (e.g. `AAPL`, `BRK.B`). Default range: `2017-01-01` to today.                                                                                                                                   |
| `--stock-list`    | Comma-separated list of tickers to download in a single batch call (e.g. `AAPL,MSFT,GOOG`).                                                                                                                                  |
| `--range N`       | When used with `--download`, restrict to the first `N` symbols of the S&P 500 list (handy for smoke tests).                                                                                                                  |
| `--update`        | Incrementally update every CSV in `--folder`. Resumes from the last date in each file unless `--start` is given.                                                                                                             |
| `--update-stock`  | Incrementally update a single ticker (pass the symbol as the value, e.g. `--update-stock=AAPL`).                                                                                                                             |
| `--start`         | Start date (`YYYY-MM-DD`). Optional for updates (defaults to `last_index + 1 day`).                                                                                                                                          |
| `--end`           | End date (`YYYY-MM-DD`). Defaults to today; the literal string `today` is also accepted.                                                                                                                                     |
| `--delete`        | Delete every file in `--folder` and exit.                                                                                                                                                                                    |
| `--delete-stock`  | Delete one specific ticker's CSV from `--folder`.                                                                                                                                                                            |
| `--folder`        | Source / destination folder (default `stock/`). Created automatically if missing.                                                                                                                                            |

### Usage examples

```bash
# Download every S&P 500 stock between 2017-01-01 and 2018-01-01
python3 data_downloader.py --download --start=2017-01-01 --end=2018-01-01

# Download Apple (AAPL) from 2017-01-01 to today, into ./stock/
python3 data_downloader.py --stock=AAPL --end=today --folder=stock/

# Download a small batch in one call
python3 data_downloader.py --stock-list=AAPL,MSFT,GOOG --start=2024-01-01 --folder=stock/

# Smoke test: first 5 S&P symbols only
python3 data_downloader.py --download --range=5 --start=2024-01-01 --end=2024-02-01

# Symbols with dots are normalized: BRK.B -> stock/BRK-B.csv
python3 data_downloader.py --stock=BRK.B --start=2024-01-01

# Incrementally update every CSV in the folder (resume from each file's last date)
python3 data_downloader.py --update --folder=stock/

# Incrementally update one ticker; second run is a no-op when up-to-date
python3 data_downloader.py --update-stock=AAPL --folder=stock/

# Delete one ticker, or wipe the folder
python3 data_downloader.py --delete-stock=AAPL --folder=stock/
python3 data_downloader.py --delete --folder=stock/
```

> **Note:** the data downloader is a standalone CLI tool — it is not imported by any `stockpy` model code. The forecasters in `stockpy.forecasters` accept any 2-D NumPy array shaped `(time_steps, n_features)`, regardless of source.

## TODOs
Planned enhancements. Contributions and suggestions are welcome.

- [x] Comprehensive `test/` suite covering every forecaster.
- [ ] Expand documentation with end-to-end forecasting tutorials and API reference.
- [ ] Add more encoder-decoder architectures (Informer, N-BEATS, PatchTST).
- [ ] Probabilistic uncertainty intervals via Monte Carlo sampling on `DMMForecaster`.
- [ ] Multi-series / panel-data training support.

*Note: ✅ / a checked box indicates the task has been completed.*

## Authors and acknowledgements
**stockpy** is currently developed and maintained by **Silvio Baratto**. Contact:
- silvio.baratto22 at gmail.com

## Reporting a bug
The best way to report a bug is via the [Issues](https://github.com/SilvioBaratto/stockpy/issues) section. Please be clear and include a minimal reproducible example (input shape, model hyperparameters, full traceback).

## How to contribute

Contributions on tests, documentation, and new features are welcome. The [Issues](https://github.com/SilvioBaratto/stockpy/issues) tracker lists open work.

Guidelines for submitting a patch:

1. Open a new [issue](https://github.com/SilvioBaratto/stockpy/issues) describing the bug or feature, so we can avoid duplicate work.
2. Fork the project and create a dedicated branch (e.g. `fix-issue-22`).
3. Run [black](https://github.com/psf/black) (88-char lines) before pushing.
4. Provide meaningful commit messages.
5. Submit your pull request.

## License

See the [LICENSE](LICENSE) file for license rights and limitations (MIT).

## stockpy Legal Disclaimer

Please read this legal disclaimer carefully before using stockpy-learn library. By using stockpy-learn library, you agree to be bound by this disclaimer.

stockpy-learn library is provided for informational and educational purposes only and is not intended as a recommendation, offer or solicitation for the purchase or sale of any financial instrument or securities. The information provided in the stockpy-learn library is not to be construed as financial, investment, legal, or tax advice, and the use of any information provided in stockpy-learn library is at your own risk.

stockpy-learn library is not a substitute for professional financial or investment advice and should not be relied upon for making investment decisions. You should consult a qualified financial or investment professional before making any investment decision.

We make no representation or warranty, express or implied, as to the accuracy, completeness, or suitability of any information provided in stockpy, and we shall not be liable for any errors or omissions in such information.

We shall not be liable for any direct, indirect, incidental, special, consequential, or exemplary damages arising from the use of stockpy library or any information provided therein.

stockpy-learn library is provided "as is" without warranty of any kind, either express or implied, including but not limited to the implied warranties of merchantability, fitness for a particular purpose, or non-infringement.

We reserve the right to modify or discontinue stockpy-learn library at any time without notice. We shall not be liable for any modification, suspension, or discontinuance of stockpy-learn library.

By using stockpy-learn library, you agree to indemnify and hold us harmless from any claim or demand, including reasonable attorneys' fees, made by any third party due to or arising out of your use of stockpy-learn library, your violation of this disclaimer, or your violation of any law or regulation.

This legal disclaimer is governed by and construed in accordance with the laws of Italy, and any disputes relating to this disclaimer shall be subject to the exclusive jurisdiction of the courts of Italy.

If you have any questions about this legal disclaimer, please contact us at silvio.baratto22@gmail.com.

By using stockpy-learn library, you acknowledge that you have read and understood this legal disclaimer and agree to be bound by its terms and conditions.
