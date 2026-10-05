import torch

from src.tokenizer import BasicTokenizer
from src.bert import BertForMaskedLM


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 1. LOAD CORPUS
# ============================================================

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


# ============================================================
# 2. REBUILD TOKENIZER
# ============================================================

tokenizer = BasicTokenizer()

tokenizer.build_vocab(texts)


# ============================================================
# 3. CREATE MODEL
# ============================================================

model = BertForMaskedLM(
    vocab_size=tokenizer.vocab_size,
    hidden_size=64,
    num_heads=4,
    num_layers=2,
    intermediate_size=256,
    max_position_embeddings=16,
    dropout=0.1,
).to(DEVICE)


# ============================================================
# 4. LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    torch.load(
        "bert_real_text.pt",
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# 5. CREATE MASKED SENTENCE
# ============================================================

text = "BERT uses bidirectional self attention."

tokens = tokenizer.tokenize(text)

print("Original tokens:")
print(tokens)


# We manually create:
#
# [CLS] BERT uses bidirectional self [MASK] .
# [SEP]

mask_token = "[MASK]"

tokens_with_mask = [
    "[CLS]",
    "bert",
    "uses",
    "bidirectional",
    "self",
    mask_token,
    ".",
    "[SEP]",
]


input_ids = [
    tokenizer.vocab.get(
        token,
        tokenizer.vocab["[UNK]"]
    )
    for token in tokens_with_mask
]


input_ids = torch.tensor(
    [input_ids],
    dtype=torch.long,
    device=DEVICE
)


attention_mask = torch.ones_like(
    input_ids
)


# ============================================================
# 6. RUN BERT
# ============================================================

with torch.no_grad():

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


logits = output["logits"]


# ============================================================
# 7. FIND [MASK]
# ============================================================

mask_id = tokenizer.vocab["[MASK]"]

mask_position = (
    input_ids[0] == mask_id
).nonzero(as_tuple=True)[0].item()


mask_logits = logits[
    0,
    mask_position
]


# ============================================================
# 8. TOP PREDICTIONS
# ============================================================

probabilities = torch.softmax(
    mask_logits,
    dim=-1
)

top_probabilities, top_indices = (
    torch.topk(
        probabilities,
        k=5
    )
)


print("\nMasked sentence:")
print(
    "BERT uses bidirectional self [MASK] ."
)


print("\nTop predictions:")

for probability, token_id in zip(
    top_probabilities,
    top_indices
):

    token = next(
        (
            token
            for token, idx
            in tokenizer.vocab.items()
            if idx == token_id.item()
        ),
        "[UNK]"
    )

    print(
        f"{token:20s} "
        f"{probability.item():.4f}"
    )