import torch

from src.bert import BertForPreTraining


def test_mlm_weights_are_tied():

    model = BertForPreTraining(
        vocab_size=100,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        intermediate_size=256,
        max_position_embeddings=16,
    )

    embedding_weight = (
        model.bert.embeddings.word_embeddings.weight
    )

    decoder_weight = (
        model.mlm_head.decoder.weight
    )

    assert embedding_weight.data_ptr() == decoder_weight.data_ptr()