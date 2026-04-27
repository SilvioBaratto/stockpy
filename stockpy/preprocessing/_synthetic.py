import numpy as np


def make_synthetic_series(
    length=500,
    n_features=3,
    freq=0.1,
    noise_std=0.05,
    seed=42,
):
    rng = np.random.RandomState(seed)
    t = np.arange(length, dtype=np.float32)
    series = np.zeros((length, n_features), dtype=np.float32)

    for i in range(n_features):
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(0.5, 1.5)
        series[:, i] = amplitude * np.sin(freq * t + phase)

    if noise_std > 0:
        noise = rng.normal(0, noise_std, (length, n_features)).astype(np.float32)
        series += noise

    return series
