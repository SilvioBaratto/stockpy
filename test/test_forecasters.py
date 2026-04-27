import pytest
import torch

from stockpy.forecasters import (
    LSTMForecaster,
    GRUForecaster,
    BiLSTMForecaster,
    BiGRUForecaster,
    TCNForecaster,
)
from stockpy.base import EncoderDecoderForecaster


class TestForecasterStubs:
    """Tests that forecaster stubs can be imported and have the expected interface."""

    @pytest.mark.parametrize("cls", [
        LSTMForecaster,
        GRUForecaster,
        BiLSTMForecaster,
        BiGRUForecaster,
        TCNForecaster,
    ])
    def test_stub_is_subclass_of_encoder_decoder_forecaster(self, cls):
        assert issubclass(cls, EncoderDecoderForecaster)

    @pytest.mark.parametrize("cls", [
        LSTMForecaster,
        GRUForecaster,
        BiLSTMForecaster,
        BiGRUForecaster,
        TCNForecaster,
    ])
    def test_stub_has_context_len_and_pred_len(self, cls):
        inst = cls(context_len=10, pred_len=5)
        assert hasattr(inst, 'context_len')
        assert hasattr(inst, 'pred_len')

    @pytest.mark.parametrize("cls", [
        LSTMForecaster,
        GRUForecaster,
        BiLSTMForecaster,
        BiGRUForecaster,
        TCNForecaster,
    ])
    def test_stub_init_accepts_context_len_and_pred_len(self, cls):
        inst = cls(context_len=10, pred_len=5)
        assert inst.context_len == 10
        assert inst.pred_len == 5

    def test_stub_forward_raises_not_implemented(self):
        # All forecasters are now implemented; this test is kept as a no-op
        # placeholder for future stubs.
        pass

    def test_lstm_forward_does_not_raise_not_implemented(self):
        from stockpy.forecasters._lstm import LSTMModel
        inst = LSTMForecaster(context_len=4, pred_len=2, rnn_size=4, hidden_size=4)
        inst.module_ = LSTMModel(
            n_features=2, pred_len=2, rnn_size=4, hidden_size=4
        )
        out = inst.forward(torch.randn(1, 4, 2))
        assert out.shape == (1, 2, 2)

    def test_gru_forward_does_not_raise_not_implemented(self):
        from stockpy.forecasters._gru import GRUModel
        inst = GRUForecaster(context_len=4, pred_len=2, rnn_size=4, hidden_size=4)
        inst.module_ = GRUModel(
            n_features=2, pred_len=2, rnn_size=4, hidden_size=4
        )
        out = inst.forward(torch.randn(1, 4, 2))
        assert out.shape == (1, 2, 2)

    def test_bilstm_forward_does_not_raise_not_implemented(self):
        from stockpy.forecasters._bilstm import BiLSTMModel
        inst = BiLSTMForecaster(context_len=4, pred_len=2, rnn_size=4, hidden_size=4)
        inst.module_ = BiLSTMModel(
            n_features=2, pred_len=2, rnn_size=4, hidden_size=4
        )
        out = inst.forward(torch.randn(1, 4, 2))
        assert out.shape == (1, 2, 2)

    def test_bigru_forward_does_not_raise_not_implemented(self):
        from stockpy.forecasters._bigru import BiGRUModel
        inst = BiGRUForecaster(context_len=4, pred_len=2, rnn_size=4, hidden_size=4)
        inst.module_ = BiGRUModel(
            n_features=2, pred_len=2, rnn_size=4, hidden_size=4
        )
        out = inst.forward(torch.randn(1, 4, 2))
        assert out.shape == (1, 2, 2)

    def test_tcn_forward_does_not_raise_not_implemented(self):
        from stockpy.forecasters._tcn import TCNModel
        inst = TCNForecaster(context_len=4, pred_len=2, num_filters=4, hidden_size=4)
        inst.module_ = TCNModel(
            n_features=2, pred_len=2, num_filters=4, hidden_size=4
        )
        out = inst.forward(torch.randn(1, 4, 2))
        assert out.shape == (1, 2, 2)

    @pytest.mark.parametrize("cls", [
        LSTMForecaster,
        GRUForecaster,
        BiLSTMForecaster,
        BiGRUForecaster,
        TCNForecaster,
    ])
    def test_stub_module_type_attribute(self, cls):
        inst = cls(context_len=4, pred_len=2)
        assert hasattr(inst, 'model_type')
        assert isinstance(inst.model_type, str)

    def test_stockpy_exports_all_cycle2_forecasters(self):
        import stockpy
        for name in [
            'LSTMForecaster',
            'GRUForecaster',
            'BiLSTMForecaster',
            'BiGRUForecaster',
            'TCNForecaster',
        ]:
            assert hasattr(stockpy, name), f"stockpy does not export {name}"
