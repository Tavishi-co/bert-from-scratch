import torch

from src.embeddings import BertEmbeddings


def test_embeddings_shape():

    model = BertEmbeddings(
        vocab_size=100,
        hidden_size=64,
        max_position_embeddings=32,
    )

    input_ids = torch.randint(
        0,
        100,
        (2, 10)
    )

    output = model(input_ids)

    assert output.shape == (2, 10, 64)