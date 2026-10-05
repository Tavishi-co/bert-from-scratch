import torch
from torch.utils.data import DataLoader

from src.bert import BertForPreTraining
from src.dataset import BertPretrainingDataset
from src.tokenizer import BasicTokenizer


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
MAX_LENGTH = 16

HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 256


# ---------------------------------------------------------
# 2. Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ---------------------------------------------------------
# 3. Load corpus
# ---------------------------------------------------------

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

print("Number of sentences:", len(texts))


# ---------------------------------------------------------
# 4. Build tokenizer
# ---------------------------------------------------------

tokenizer = BasicTokenizer()

tokenizer.build_vocab(texts)

print("Vocabulary size:", tokenizer.vocab_size)


# ---------------------------------------------------------
# 5. Create pretraining dataset
# ---------------------------------------------------------

dataset = BertPretrainingDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# ---------------------------------------------------------
# 6. Create BERT
# ---------------------------------------------------------

model = BertForPreTraining(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_LENGTH,
    dropout=0.1,
)

model = model.to(device)


# ---------------------------------------------------------
# 7. Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ---------------------------------------------------------
# 8. Training
# ---------------------------------------------------------

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0
    total_mlm_loss = 0.0
    total_nsp_loss = 0.0

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        mlm_labels = batch["mlm_labels"].to(device)
        nsp_labels = batch["nsp_labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            mlm_labels=mlm_labels,
            nsp_labels=nsp_labels,
        )

        loss = outputs["loss"]

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_mlm_loss += outputs["mlm_loss"].item()
        total_nsp_loss += outputs["nsp_loss"].item()

    avg_loss = total_loss / len(dataloader)
    avg_mlm_loss = total_mlm_loss / len(dataloader)
    avg_nsp_loss = total_nsp_loss / len(dataloader)

    if (
        epoch == 0
        or (epoch + 1) % 10 == 0
    ):
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Total Loss: {avg_loss:.4f} | "
            f"MLM Loss: {avg_mlm_loss:.4f} | "
            f"NSP Loss: {avg_nsp_loss:.4f}"
        )


# ---------------------------------------------------------
# 9. Save model
# ---------------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "max_position_embeddings": MAX_LENGTH,
        },
    },
    "results/bert_pretrained.pt",
)

print("\nTraining complete.")
print("Model saved to results/bert_pretrained.pt")