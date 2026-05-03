import numpy as np
import torch

from stockpy.forecasters import DMMForecaster
from stockpy.base import EncoderDecoderForecaster


class TestDMMForecaster:
    """Comprehensive tests for DMMForecaster probabilistic encoder-decoder."""

    def test_is_subclass_of_encoder_decoder_forecaster(self):
        assert issubclass(DMMForecaster, EncoderDecoderForecaster)

    def test_instantiation_sets_context_and_pred_len(self):
        model = DMMForecaster(context_len=10, pred_len=5)
        assert model.context_len == 10
        assert model.pred_len == 5

    def test_instantiation_with_all_parameters(self):
        model = DMMForecaster(
            z_dim=16,
            emission_dim=16,
            transition_dim=16,
            rnn_dim=16,
            num_layers=1,
            dropout=0.1,
            variance=0.05,
            activation="tanh",
            bias=True,
            context_len=10,
            pred_len=5,
        )
        assert model.z_dim == 16
        assert model.emission_dim == 16
        assert model.transition_dim == 16
        assert model.rnn_dim == 16
        assert model.num_layers == 1
        assert model.dropout == 0.1
        assert model.variance == 0.05
        assert model.activation == "tanh"
        assert model.bias is True

    def test_model_type_attribute(self):
        model = DMMForecaster(context_len=4, pred_len=2)
        assert model.model_type == "rnn"

    def test_prob_flag_is_true(self):
        model = DMMForecaster(context_len=4, pred_len=2)
        assert model.prob is True

    def test_initialize_module_creates_module_(self):
        model = DMMForecaster(context_len=4, pred_len=2, z_dim=8, rnn_dim=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        assert hasattr(model, "module_")
        assert isinstance(model.module_, torch.nn.Module)

    def test_forward_output_shape(self):
        model = DMMForecaster(context_len=4, pred_len=2, z_dim=8, rnn_dim=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)

        x_batch = torch.randn(2, 4, 3)
        out = model.forward(x_batch)
        assert out.shape == (2, 2, 3)

    def test_predict_returns_correct_shape(self):
        model = DMMForecaster(
            context_len=10, pred_len=5, z_dim=16, rnn_dim=16, num_layers=1
        )
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)

        X_test = torch.randn(5, 10, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(5, 1))
        preds = model.predict(ds)
        assert preds.shape == (5, 5, 3)

    def test_fit_5_epochs(self):
        from stockpy.preprocessing._synthetic import make_synthetic_series

        series = make_synthetic_series(length=200, n_features=3)
        X = series
        y = series
        model = DMMForecaster(
            context_len=20, pred_len=5, z_dim=16, rnn_dim=16, num_layers=1
        )
        model.fit(X, y, epochs=5, verbose=0, train_split=None)
        assert model.initialized_
        assert len(model.history) == 5

    def test_svi_is_initialized(self):
        model = DMMForecaster(context_len=4, pred_len=2, z_dim=8, rnn_dim=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        assert hasattr(model, "svi_")
        assert model.svi_ is not None

    def test_elbo_is_trace_mean_field(self):
        from pyro.infer import TraceMeanField_ELBO

        model = DMMForecaster(context_len=4, pred_len=2, z_dim=8, rnn_dim=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        assert hasattr(model, "elbo")
        assert isinstance(model.elbo, TraceMeanField_ELBO)

    def test_encoder_processes_context(self):
        model = DMMForecaster(context_len=4, pred_len=2, z_dim=8, rnn_dim=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        assert hasattr(model.module_, "rnn")

    def test_decoder_produces_future(self):
        model = DMMForecaster(context_len=4, pred_len=2, z_dim=8, rnn_dim=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        assert hasattr(model.module_, "transition")
        assert hasattr(model.module_, "emitter")

    def test_save_load_roundtrip(self, tmp_path):
        model = DMMForecaster(
            context_len=10, pred_len=5, z_dim=8, rnn_dim=8, num_layers=1
        )
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)

        f_params = str(tmp_path / "params.pt")
        model.save_params(f_params=f_params)

        X_test = torch.randn(5, 10, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(5, 1))
        preds_before = model.predict(ds)

        model2 = DMMForecaster(
            context_len=10, pred_len=5, z_dim=8, rnn_dim=8, num_layers=1
        )
        model2.fit(X, y, epochs=0, verbose=0, train_split=None)
        model2.load_params(f_params=f_params)

        preds_after = model2.predict(ds)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

    def test_forward_does_not_raise_not_implemented(self):
        model = DMMForecaster(context_len=4, pred_len=2, z_dim=8, rnn_dim=8)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 3).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        x_batch = torch.randn(1, 4, 3)
        out = model.forward(x_batch)
        assert isinstance(out, torch.Tensor)

    def test_no_classifier_logic(self):
        import stockpy.forecasters._dmm as dmm_module

        source = dmm_module.__file__
        with open(source, "r") as f:
            code = f.read()
        assert "Classifier" not in code
        assert "CrossEntropyLoss" not in code
        assert "accuracy_score" not in code

    def test_on_synthetic_data_with_early_stopping(self):
        from stockpy.callbacks import EarlyStopping
        from stockpy.preprocessing._synthetic import make_synthetic_series

        series = make_synthetic_series(length=200, n_features=3)
        X = series
        y = series
        model = DMMForecaster(
            context_len=20,
            pred_len=5,
            z_dim=16,
            rnn_dim=16,
            num_layers=1,
        )
        model.fit(
            X,
            y,
            epochs=10,
            verbose=0,
            train_split=None,
            callbacks=[
                EarlyStopping(
                    monitor="train_loss",
                    patience=1,
                    threshold=1000,
                    threshold_mode="abs",
                    lower_is_better=True,
                )
            ],
        )
        assert len(model.history) < 10

    def test_predict_returns_numpy_array(self):
        model = DMMForecaster(
            context_len=10, pred_len=5, z_dim=16, rnn_dim=16, num_layers=1
        )
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0, train_split=None)

        X_test = torch.randn(5, 10, 3)
        ds = torch.utils.data.TensorDataset(X_test, torch.zeros(5, 1))
        preds = model.predict(ds)
        assert isinstance(preds, np.ndarray)

    def test_context_and_pred_len_used_in_dataset(self):
        model = DMMForecaster(context_len=8, pred_len=3)
        X = np.random.randn(50, 2).astype(np.float32)
        y = np.random.randn(50, 2).astype(np.float32)
        model.fit(X, y, epochs=0, verbose=0, train_split=None)
        ds = model.get_dataset(X, y)
        xi, yi = ds[0]
        assert xi.shape == (8, 2)
        assert yi.shape == (3, 2)
