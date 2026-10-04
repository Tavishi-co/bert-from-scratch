import torch

from src.masking import mask_tokens


def test_masking_shapes():

    input_ids = torch.randint(
        5,
        30,
        (4, 20)
    )

    masked, labels = mask_tokens(
        input_ids,
        mask_token_id=3,
        vocab_size=30,
    )

    assert masked.shape == input_ids.shape
    assert labels.shape == input_ids.shape


def test_masked_positions_have_labels():

    input_ids = torch.randint(
        5,
        30,
        (8, 20)
    )

    _, labels = mask_tokens(
        input_ids,
        mask_token_id=3,
        vocab_size=30,
    )

    assert (labels != -100).any()