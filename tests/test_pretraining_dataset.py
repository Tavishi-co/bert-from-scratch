from src.dataset import BertPretrainingDataset
from src.tokenizer import BasicTokenizer


def test_pretraining_dataset():

    texts = [
        "BERT uses self attention.",
        "Transformers process sequences.",
        "Language models learn representations.",
        "Attention connects tokens.",
    ]

    tokenizer = BasicTokenizer()

    tokenizer.build_vocab(texts)

    dataset = BertPretrainingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=16,
    )

    example = dataset[0]

    assert example["input_ids"].shape == (16,)

    assert example["token_type_ids"].shape == (16,)

    assert example["attention_mask"].shape == (16,)

    assert example["mlm_labels"].shape == (16,)

    assert example["nsp_labels"].shape == ()

    assert example["nsp_labels"].item() in [0, 1]