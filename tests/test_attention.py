import torch

from src.attention import MultiHeadSelfAttention


def test_attention_shape():

    model = MultiHeadSelfAttention(
        hidden_size=64,
        num_heads=4,
    )

    x = torch.randn(
        2,
        10,
        64
    )

    output, weights = model(x)

    assert output.shape == (2, 10, 64)

    assert weights.shape == (
        2,
        4,
        10,
        10
    )


def test_attention_is_bidirectional():

    model = MultiHeadSelfAttention(
        hidden_size=64,
        num_heads=4,
    )

    x = torch.randn(
        1,
        5,
        64
    )

    _, weights = model(x)

    # Token 0 is allowed to attend
    # to token 4.
    assert weights[0, 0, 0, 4] > 0