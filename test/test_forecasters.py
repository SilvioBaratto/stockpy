import pytest
import torch

from stockpy.forecasters import (
    LSTMForecaster,
    GRUForecaster,
    BiLSTMForecaster,
    BiGRUForecaster,
    TCNForecaster,
    TransformerForecaster,
    DMMForecaster,
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
        TransformerForecaster,
        DMMForecaster,
    ])
    def test_stub_is_subclass_of_encoder_decoder_forecaster(self, cls):
        assert issubclass(cls, EncoderDecoderForecaster)

    @pytest.mark.parametrize("cls", [
        LSTMForecaster,
        GRUForecaster,
        BiLSTMForecaster,
        BiGRUForecaster,
        TCNForecaster,
        TransformerForecaster,
        DMMForecaster,
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
        TransformerForecaster,
        DMMForecaster,
    ])
    def test_stub_init_accepts_context_len_and_pred_len(self, cls):
        # DMMForecaster does not accept context_len/pred_len directly since it inherits
        # from the EncoderDecoderForecaster base class which handles these.
        if cls is DMMForecaster:
            inst = cls(context_len=10, pred_len=5)
        else:
            inst = cls(context_len=10, pred_len=5)
        assert inst.context_len == 10
        assert inst.pred_len == 5

    @pytest.mark.parametrize("cls", [
        LSTMForecaster,
        GRUForecaster,
        BiLSTMForecaster,
        BiGRUForecaster,
        TCNForecaster,
        TransformerForecaster,
        DMMForecaster,
    ])
    def test_stub_forward_raises_not_implemented(self, cls):
        inst = cls(context_len=4, pred_len=2)
        with pytest.raises(NotImplementedError):
            inst.forward(torch.randn(1, 4, 2))

    @pytest.mark.parametrize("cls", [
        LSTMForecaster,
        GRUForecaster,
        BiLSTMForecaster,
        BiGRUForecaster,
        TCNForecaster,
        TransformerForecaster,
        DMMForecaster,
    ])
    def test_stub_module_type_attribute(self, cls):
        inst = cls(context_len=4, pred_len=2)
        assert hasattr(inst, 'model_type')
        assert isinstance(inst.model_type, str)

    def test_stockpy_exports_all_stubs(self):
        import stockpy
        for name in [
            'LSTMForecaster',
            'GRUForecaster',
            'BiLSTMForecaster',
            'BiGRUForecaster',
            'TCNForecaster',
            'TransformerForecaster',
            'DMMForecaster',
        ]:
            assert hasattr(stockpy, name), f"stockpy does not export {name}"
