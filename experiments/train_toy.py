import torch

from src.bert import BertForMaskedLM
from src.masking import mask_tokens


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VOCAB_SIZE = 50
SEQ_LEN = 16

HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 256

BATCH_SIZE = 32
STEPS = 1000
LEARNING_RATE = 3e-4

PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
MASK_ID = 3


# --------------------------------------------------
# Model
# --------------------------------------------------

model = BertForMaskedLM(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=SEQ_LEN,
).to(DEVICE)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# Training
# --------------------------------------------------

model.train()

for step in range(1, STEPS + 1):

    # Generate a batch of toy token sequences.
    input_ids = torch.randint(
        5,
        VOCAB_SIZE,
        (BATCH_SIZE, SEQ_LEN),
        device=DEVICE,
    )

    # Apply BERT's MLM masking strategy.
    masked_input_ids, labels = mask_tokens(
        input_ids.clone(),
        mask_token_id=MASK_ID,
        vocab_size=VOCAB_SIZE,
        special_token_ids=[
            PAD_ID,
            CLS_ID,
            SEP_ID,
        ],
    )

    # No padding in this toy experiment.
    attention_mask = torch.ones_like(
        masked_input_ids
    )

    # Clear old gradients.
    optimizer.zero_grad()

    # Forward pass.
    output = model(
        input_ids=masked_input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = output["loss"]

    # Backpropagation.
    loss.backward()

    # Update parameters.
    optimizer.step()

    if step % 100 == 0:

        print(
            f"Step {step:4d} | "
            f"Loss {loss.item():.4f}"
        )


# --------------------------------------------------
# Save model
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "bert_toy.pt"
)

print("\nTraining complete.")