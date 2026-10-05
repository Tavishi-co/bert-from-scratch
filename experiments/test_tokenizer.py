from src.tokenizer import BasicTokenizer


# Read our small text corpus
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


# Create tokenizer
tokenizer = BasicTokenizer()


# Build vocabulary from corpus
tokenizer.build_vocab(texts)


print("Vocabulary size:", tokenizer.vocab_size)


# Test one sentence
text = texts[0]

print("\nOriginal:")
print(text)

print("\nTokens:")
print(tokenizer.tokenize(text))

print("\nEncoded:")
print(
    tokenizer.encode(
        text,
        max_length=16
    )
)