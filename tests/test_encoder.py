import torch

from src.encoder import BertEncoder


def test_encoder_shape():

    model = BertEncoder(
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        intermediate_size=256,
    )

    x = torch.randn(
        2,
        10,
        64
    )

    output, attention_weights = model(x)

    assert output.shape == (2, 10, 64)

    assert len(attention_weights) == 2