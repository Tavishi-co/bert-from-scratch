import torch

from src.bert import BertForMaskedLM


def test_bert_forward():

    model = BertForMaskedLM(
        vocab_size=50,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        intermediate_size=256,
        max_position_embeddings=32,
    )

    input_ids = torch.randint(
        0,
        50,
        (2, 12)
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    labels = input_ids.clone()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert output["logits"].shape == (
        2,
        12,
        50
    )

    assert output["loss"] is not None