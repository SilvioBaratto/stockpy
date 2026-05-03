"""Tests for ValidSplit error handling on small datasets.

Regression coverage for issue #2: a default-configured forecaster fitted
on a series too short to yield 5 sliding windows must surface a clear,
actionable error instead of sklearn's cryptic
``Cannot have number of splits n_splits=5 greater than the number of
samples: n_samples=4``.
"""

import numpy as np
import pytest

from stockpy.preprocessing import TimeSeriesDataset, ValidSplit


def _series(n_steps, n_features=2):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_steps, n_features)).astype(np.float32)


class TestValidSplitTooFewSamples:
    """ValidSplit must reject n_splits > n_samples with an actionable error."""

    def test_when_n_splits_exceeds_n_samples_raises_value_error(self):
        dataset = TimeSeriesDataset(_series(24), context_len=20, pred_len=1)
        assert len(dataset) == 4

        with pytest.raises(ValueError):
            ValidSplit(cv=5)(dataset)

    def test_when_n_splits_exceeds_n_samples_error_mentions_both_counts(self):
        dataset = TimeSeriesDataset(_series(24), context_len=20, pred_len=1)

        with pytest.raises(ValueError, match=r"4.*5|5.*4"):
            ValidSplit(cv=5)(dataset)

    def test_when_n_splits_exceeds_n_samples_error_suggests_remediation(self):
        dataset = TimeSeriesDataset(_series(24), context_len=20, pred_len=1)

        with pytest.raises(ValueError, match=r"train_split|cv|context_len"):
            ValidSplit(cv=5)(dataset)

    def test_when_n_splits_equals_n_samples_does_not_raise(self):
        dataset = TimeSeriesDataset(_series(25), context_len=20, pred_len=1)
        assert len(dataset) == 5

        train, valid = ValidSplit(cv=5)(dataset)
        assert len(train) + len(valid) == 5

    def test_when_n_splits_below_n_samples_returns_split(self):
        dataset = TimeSeriesDataset(_series(40), context_len=20, pred_len=1)
        train, valid = ValidSplit(cv=5)(dataset)
        assert len(train) > 0
        assert len(valid) > 0

    def test_when_cv_is_float_holdout_works_on_small_dataset(self):
        """A float cv (holdout fraction) must still work where int cv fails."""
        dataset = TimeSeriesDataset(_series(24), context_len=20, pred_len=1)

        train, valid = ValidSplit(cv=0.25)(dataset)
        assert len(train) + len(valid) == 4

    def test_when_cv_is_two_on_dataset_of_four_works(self):
        """Smaller int cv values must not be affected by the fix."""
        dataset = TimeSeriesDataset(_series(24), context_len=20, pred_len=1)

        train, valid = ValidSplit(cv=2)(dataset)
        assert len(train) + len(valid) == 4
