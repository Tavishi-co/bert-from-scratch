import torch
from torch.utils.data import DataLoader

from src.bert import BertForPreTraining
from src.dataset import BertPretrainingDataset
from src.tokenizer import BasicTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_LENGTH = 16

HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 256

BATCH_SIZE = 4


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ---------------------------------------------------------
# Load corpus
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


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

tokenizer = BasicTokenizer()
tokenizer.build_vocab(texts)

print("Vocabulary size:", tokenizer.vocab_size)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

dataset = BertPretrainingDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = BertForPreTraining(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_LENGTH,
)

checkpoint = torch.load(
    "results/bert_pretrained.pt",
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

total_loss = 0.0
total_mlm_loss = 0.0
total_nsp_loss = 0.0

total_masked_tokens = 0
correct_mlm_predictions = 0

total_nsp_examples = 0
correct_nsp_predictions = 0


with torch.no_grad():

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        mlm_labels = batch["mlm_labels"].to(device)
        nsp_labels = batch["nsp_labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            mlm_labels=mlm_labels,
            nsp_labels=nsp_labels,
        )

        total_loss += outputs["loss"].item()
        total_mlm_loss += outputs["mlm_loss"].item()
        total_nsp_loss += outputs["nsp_loss"].item()

        # -------------------------------------------------
        # MLM accuracy
        # -------------------------------------------------

        predictions = outputs["mlm_logits"].argmax(dim=-1)

        mask = mlm_labels != -100

        correct_mlm_predictions += (
            predictions[mask] == mlm_labels[mask]
        ).sum().item()

        total_masked_tokens += mask.sum().item()

        # -------------------------------------------------
        # NSP accuracy
        # -------------------------------------------------

        nsp_predictions = outputs["nsp_logits"].argmax(dim=-1)

        correct_nsp_predictions += (
            nsp_predictions == nsp_labels
        ).sum().item()

        total_nsp_examples += nsp_labels.size(0)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

num_batches = len(dataloader)

avg_loss = total_loss / num_batches
avg_mlm_loss = total_mlm_loss / num_batches
avg_nsp_loss = total_nsp_loss / num_batches

mlm_accuracy = (
    correct_mlm_predictions / total_masked_tokens
    if total_masked_tokens > 0
    else 0.0
)

nsp_accuracy = (
    correct_nsp_predictions / total_nsp_examples
    if total_nsp_examples > 0
    else 0.0
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n========== BERT PRETRAINING EVALUATION ==========")

print(f"Total Loss:     {avg_loss:.4f}")
print(f"MLM Loss:       {avg_mlm_loss:.4f}")
print(f"NSP Loss:       {avg_nsp_loss:.4f}")

print()

print(
    f"MLM Accuracy:   {mlm_accuracy * 100:.2f}%"
)

print(
    f"NSP Accuracy:   {nsp_accuracy * 100:.2f}%"
)

print()

print("Masked tokens:", total_masked_tokens)
print("NSP examples:", total_nsp_examples)

print("==================================================")