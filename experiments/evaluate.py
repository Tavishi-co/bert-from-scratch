import torch
from torch.utils.data import DataLoader

from src.tokenizer import BasicTokenizer
from src.dataset import BertTextDataset
from src.bert import BertForMaskedLM


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Load corpus

with open(
    "experiments/corpus.txt",
    "r",
    encoding="utf-8"
) as f:

    texts = [
        line.strip()
        for line in f
        if line.strip()
    ]


# Tokenizer

tokenizer = BasicTokenizer()

tokenizer.build_vocab(texts)


# Dataset

dataset = BertTextDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_length=16,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
)


# Model

model = BertForMaskedLM(
    vocab_size=tokenizer.vocab_size,
    hidden_size=64,
    num_heads=4,
    num_layers=2,
    intermediate_size=256,
    max_position_embeddings=16,
).to(DEVICE)


model.load_state_dict(
    torch.load(
        "bert_real_text.pt",
        map_location=DEVICE
    )
)

model.eval()


total_loss = 0.0
correct = 0
total = 0


with torch.no_grad():

    for batch in loader:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_loss += output["loss"].item()

        predictions = output[
            "logits"
        ].argmax(dim=-1)

        mask = labels != -100

        correct += (
            predictions[mask]
            == labels[mask]
        ).sum().item()

        total += mask.sum().item()


average_loss = (
    total_loss / len(loader)
)

accuracy = (
    correct / total
    if total > 0
    else 0.0
)


print(
    f"Evaluation Loss: {average_loss:.4f}"
)

print(
    f"MLM Accuracy: {accuracy * 100:.2f}%"
)