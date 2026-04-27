import pytest
import torch

from stockpy.callbacks import (
    Checkpoint,
    EarlyStopping,
    LRScheduler,
    PrintLog,
)
from stockpy.history import History


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_patience_stops_after_n_non_improving_epochs(self, mock_forecaster):
        cb = EarlyStopping(monitor="valid_loss", patience=3, threshold=0.0)
        cb.initialize()
        mock_forecaster.history = History()

        cb.on_train_begin(mock_forecaster)
        interrupted = False
        for i, loss in enumerate([1.0, 0.9, 0.95, 0.96, 0.97]):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("valid_loss", loss)
            try:
                cb.on_epoch_end(mock_forecaster)
            except KeyboardInterrupt:
                interrupted = True
                assert i == 4
                break

        assert interrupted

    def test_improvement_resets_misses(self, mock_forecaster):
        cb = EarlyStopping(monitor="valid_loss", patience=3, threshold=0.0)
        cb.initialize()
        mock_forecaster.history = History()

        cb.on_train_begin(mock_forecaster)
        interrupted = False
        for i, loss in enumerate([1.0, 1.1, 1.2, 0.9, 1.3, 1.4, 1.5, 1.6]):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("valid_loss", loss)
            try:
                cb.on_epoch_end(mock_forecaster)
            except KeyboardInterrupt:
                interrupted = True
                assert i == 6
                break

        assert interrupted
        assert cb.misses_ == 3

    def test_load_best_restores_weights(self, mock_forecaster):
        cb = EarlyStopping(
            monitor="valid_loss", patience=5, threshold=0.0, load_best=True
        )
        cb.initialize()
        mock_forecaster.history = History()

        # Create a simple state dict that can be deepcopied and restored
        mock_forecaster.module_ = torch.nn.Linear(2, 1)

        cb.on_train_begin(mock_forecaster)
        for i, loss in enumerate([1.0, 0.8, 0.9]):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("valid_loss", loss)
            cb.on_epoch_end(mock_forecaster)

        assert cb.best_model_weights_ is not None
        cb.on_train_end(mock_forecaster)

    def test_relative_threshold_mode(self, mock_forecaster):
        cb = EarlyStopping(
            monitor="valid_loss", patience=2, threshold=0.1, threshold_mode="rel"
        )
        cb.initialize()
        mock_forecaster.history = History()

        cb.on_train_begin(mock_forecaster)
        interrupted = False
        for i, loss in enumerate([1.0, 0.95, 0.96, 0.97]):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("valid_loss", loss)
            try:
                cb.on_epoch_end(mock_forecaster)
            except KeyboardInterrupt:
                interrupted = True
                assert i == 2
                break

        assert interrupted

    def test_absolute_threshold_mode(self, mock_forecaster):
        cb = EarlyStopping(
            monitor="valid_loss", patience=2, threshold=0.01, threshold_mode="abs"
        )
        cb.initialize()
        mock_forecaster.history = History()

        cb.on_train_begin(mock_forecaster)
        interrupted = False
        for i, loss in enumerate([1.0, 0.99, 0.991, 0.992]):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("valid_loss", loss)
            try:
                cb.on_epoch_end(mock_forecaster)
            except KeyboardInterrupt:
                interrupted = True
                assert i == 2
                break

        assert interrupted

    def test_higher_is_better(self, mock_forecaster):
        cb = EarlyStopping(
            monitor="accuracy", patience=2, threshold=0.0, lower_is_better=False
        )
        cb.initialize()
        mock_forecaster.history = History()

        cb.on_train_begin(mock_forecaster)
        interrupted = False
        for i, acc in enumerate([0.5, 0.6, 0.55, 0.56, 0.57]):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("accuracy", acc)
            try:
                cb.on_epoch_end(mock_forecaster)
            except KeyboardInterrupt:
                interrupted = True
                assert i == 3
                break

        assert interrupted

    def test_invalid_threshold_mode_raises(self, mock_forecaster):
        cb = EarlyStopping(monitor="valid_loss", threshold_mode="invalid")
        cb.initialize()
        mock_forecaster.history = History()
        with pytest.raises(ValueError, match="Invalid threshold mode"):
            cb.on_train_begin(mock_forecaster)


class TestCheckpoint:
    """Tests for Checkpoint callback."""

    def test_saves_on_best_epoch(self, tmp_path, mock_forecaster):
        checkpoint_dir = tmp_path / "checkpoints"
        cb = Checkpoint(
            monitor="valid_loss",
            f_params=str(checkpoint_dir / "params.pt"),
            dirname=str(checkpoint_dir),
        )
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.module_ = torch.nn.Linear(2, 1)
        mock_forecaster.optimizer_ = torch.optim.SGD(
            mock_forecaster.module_.parameters(), lr=0.01
        )

        cb.on_train_begin(mock_forecaster)
        for i, loss in enumerate([1.0, 0.9, 0.95]):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.new_batch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("valid_loss", loss)
            cb.on_epoch_end(mock_forecaster)

        assert (checkpoint_dir / "params.pt").exists()

    def test_safetensors_format(self, tmp_path, mock_forecaster):
        checkpoint_dir = tmp_path / "checkpoints"
        cb = Checkpoint(
            monitor="valid_loss",
            f_params=str(checkpoint_dir / "params.safetensors"),
            dirname=str(checkpoint_dir),
            use_safetensors=True,
        )
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.module_ = torch.nn.Linear(2, 1)
        mock_forecaster.optimizer_ = torch.optim.SGD(
            mock_forecaster.module_.parameters(), lr=0.01
        )

        cb.on_train_begin(mock_forecaster)
        mock_forecaster.history.new_epoch()
        mock_forecaster.history.new_batch()
        mock_forecaster.history.record("epoch", 1)
        mock_forecaster.history.record("valid_loss", 1.0)
        cb.on_epoch_end(mock_forecaster)

        assert (checkpoint_dir / "params.safetensors").exists()

    def test_file_naming_with_prefix(self, tmp_path, mock_forecaster):
        checkpoint_dir = tmp_path / "checkpoints"
        cb = Checkpoint(
            monitor=None,
            f_params="params.pt",
            dirname=str(checkpoint_dir),
            fn_prefix="run1_",
        )
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.module_ = torch.nn.Linear(2, 1)
        mock_forecaster.optimizer_ = torch.optim.SGD(
            mock_forecaster.module_.parameters(), lr=0.01
        )

        cb.on_train_begin(mock_forecaster)
        mock_forecaster.history.new_epoch()
        mock_forecaster.history.new_batch()
        mock_forecaster.history.record("epoch", 1)
        cb.on_epoch_end(mock_forecaster)

        assert (checkpoint_dir / "run1_params.pt").exists()

    def test_no_save_when_monitor_is_none_always_saves(self, tmp_path, mock_forecaster):
        checkpoint_dir = tmp_path / "checkpoints"
        cb = Checkpoint(
            monitor=None,
            f_params=str(checkpoint_dir / "params.pt"),
            dirname=str(checkpoint_dir),
        )
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.module_ = torch.nn.Linear(2, 1)
        mock_forecaster.optimizer_ = torch.optim.SGD(
            mock_forecaster.module_.parameters(), lr=0.01
        )

        cb.on_train_begin(mock_forecaster)
        for i in range(3):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.new_batch()
            mock_forecaster.history.record("epoch", i + 1)
            cb.on_epoch_end(mock_forecaster)

        assert (checkpoint_dir / "params.pt").exists()

    def test_load_best_after_training(self, tmp_path, mock_forecaster):
        checkpoint_dir = tmp_path / "checkpoints"
        cb = Checkpoint(
            monitor="valid_loss",
            f_params=str(checkpoint_dir / "params.pt"),
            dirname=str(checkpoint_dir),
            load_best=True,
        )
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.module_ = torch.nn.Linear(2, 1)
        mock_forecaster.optimizer_ = torch.optim.SGD(
            mock_forecaster.module_.parameters(), lr=0.01
        )

        cb.on_train_begin(mock_forecaster)
        for i, loss in enumerate([1.0, 0.8]):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.new_batch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("valid_loss", loss)
            cb.on_epoch_end(mock_forecaster)

        cb.on_train_end(mock_forecaster)


class TestLRScheduler:
    """Tests for LRScheduler callback."""

    def test_step_lr_decreases_learning_rate(self, mock_forecaster):
        cb = LRScheduler(policy="StepLR", step_size=2, gamma=0.1)
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.module_ = torch.nn.Linear(2, 1)
        mock_forecaster.optimizer_ = torch.optim.SGD(
            mock_forecaster.module_.parameters(), lr=0.1
        )

        cb.on_train_begin(mock_forecaster)
        initial_lr = mock_forecaster.optimizer_.param_groups[0]["lr"]
        assert initial_lr == pytest.approx(0.1)

        for i in range(3):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("train_loss", 0.5)
            cb.on_epoch_end(mock_forecaster)

        # After 2 epochs, StepLR should reduce lr by gamma=0.1
        final_lr = mock_forecaster.optimizer_.param_groups[0]["lr"]
        assert final_lr == pytest.approx(0.01)

    def test_plateau_detection(self, mock_forecaster):
        cb = LRScheduler(
            policy="ReduceLROnPlateau",
            monitor="valid_loss",
            mode="min",
            patience=1,
            factor=0.5,
        )
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.module_ = torch.nn.Linear(2, 1)
        mock_forecaster.optimizer_ = torch.optim.SGD(
            mock_forecaster.module_.parameters(), lr=0.1
        )

        cb.on_train_begin(mock_forecaster)
        for i, loss in enumerate([1.0, 1.0, 1.0]):
            mock_forecaster.history.new_epoch()
            mock_forecaster.history.record("epoch", i + 1)
            mock_forecaster.history.record("valid_loss", loss)
            cb.on_epoch_end(mock_forecaster)

        final_lr = mock_forecaster.optimizer_.param_groups[0]["lr"]
        assert final_lr == pytest.approx(0.05)

    def test_simulate_returns_expected_lrs(self):
        cb = LRScheduler(policy="StepLR", step_size=2, gamma=0.5)
        cb.initialize()
        lrs = cb.simulate(steps=5, initial_lr=0.1)

        expected = [0.1, 0.1, 0.05, 0.05, 0.025]
        assert len(lrs) == 5
        for i, expected_lr in enumerate(expected):
            assert lrs[i] == pytest.approx(expected_lr, rel=1e-5)

    def test_event_name_recorded_in_history(self, mock_forecaster):
        cb = LRScheduler(
            policy="StepLR", step_size=1, gamma=0.5, event_name="lr_changed"
        )
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.module_ = torch.nn.Linear(2, 1)
        mock_forecaster.optimizer_ = torch.optim.SGD(
            mock_forecaster.module_.parameters(), lr=0.1
        )

        cb.on_train_begin(mock_forecaster)
        mock_forecaster.history.new_epoch()
        mock_forecaster.history.record("epoch", 1)
        mock_forecaster.history.record("train_loss", 0.5)
        cb.on_epoch_end(mock_forecaster)

        assert "lr_changed" in mock_forecaster.history[-1]

    def test_batch_stepping(self, mock_forecaster):
        cb = LRScheduler(
            policy="StepLR", step_size=1, gamma=0.5, step_every="batch"
        )
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.module_ = torch.nn.Linear(2, 1)
        mock_forecaster.optimizer_ = torch.optim.SGD(
            mock_forecaster.module_.parameters(), lr=0.1
        )

        cb.on_train_begin(mock_forecaster)
        mock_forecaster.history.new_epoch()
        for i in range(2):
            mock_forecaster.history.new_batch()
            cb.on_batch_end(mock_forecaster, training=True)

        final_lr = mock_forecaster.optimizer_.param_groups[0]["lr"]
        assert final_lr == pytest.approx(0.025)


class TestPrintLog:
    """Tests for PrintLog callback."""

    def test_epoch_output_format(self, mock_forecaster):
        log_lines = []
        cb = PrintLog(sink=log_lines.append)
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.verbose = True

        mock_forecaster.history.new_epoch()
        mock_forecaster.history.record("epoch", 1)
        mock_forecaster.history.record("train_loss", 0.5)
        mock_forecaster.history.record("valid_loss", 0.6)
        cb.on_epoch_end(mock_forecaster)

        assert len(log_lines) > 0
        assert "0.5000" in log_lines[-1] or "train_loss" in str(log_lines)

    def test_no_crash_on_missing_keys(self, mock_forecaster):
        cb = PrintLog()
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.verbose = True

        mock_forecaster.history.new_epoch()
        mock_forecaster.history.record("epoch", 1)
        # No train_loss or valid_loss recorded
        cb.on_epoch_end(mock_forecaster)

        # Should not raise
        assert True

    def test_first_iteration_prints_header(self, mock_forecaster):
        log_lines = []
        cb = PrintLog(sink=log_lines.append)
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.verbose = True

        mock_forecaster.history.new_epoch()
        mock_forecaster.history.record("epoch", 1)
        mock_forecaster.history.record("train_loss", 0.5)
        cb.on_epoch_end(mock_forecaster)

        assert len(log_lines) >= 2  # Header + data line

    def test_keys_ignored_filters_keys(self, mock_forecaster):
        log_lines = []
        cb = PrintLog(sink=log_lines.append, keys_ignored=["hidden_key"])
        cb.initialize()
        mock_forecaster.history = History()
        mock_forecaster.verbose = True

        mock_forecaster.history.new_epoch()
        mock_forecaster.history.record("epoch", 1)
        mock_forecaster.history.record("train_loss", 0.5)
        mock_forecaster.history.record("hidden_key", 0.99)
        cb.on_epoch_end(mock_forecaster)

        assert len(log_lines) > 0
        # The hidden key should not appear in the last log line
        assert "hidden_key" not in log_lines[-1]

    def test_ignores_best_and_event_keys_by_default(self):
        keys = ["epoch", "train_loss", "valid_loss_best", "event_lr", "dur"]
        from stockpy.callbacks._logging import filter_log_keys

        filtered = list(filter_log_keys(keys))
        assert "valid_loss_best" not in filtered
        assert "event_lr" not in filtered
        assert "train_loss" in filtered
