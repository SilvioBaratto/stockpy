# Examples

Runnable examples for **stockpy ≥ 0.4.0** (encoder-decoder forecasting).

## Prerequisites

From the repository root:

```bash
pip install -e ".[dev]" jupyter
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`quickstart.ipynb`](quickstart.ipynb) | End-to-end `LSTMForecaster` on the bundled `stock/AAPL.csv`: load → scale → train → forecast → plot → safetensors save/load. |

## Run

```bash
jupyter lab examples/quickstart.ipynb
# or, headless:
jupyter nbconvert --to notebook --execute examples/quickstart.ipynb --output executed.ipynb
```
