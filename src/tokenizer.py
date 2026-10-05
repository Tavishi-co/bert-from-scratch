import re


class BasicTokenizer:

    def __init__(self):
        self.vocab = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[MASK]": 3,
            "[UNK]": 4,
        }

        self.next_id = 5

    def tokenize(self, text):

        text = text.lower()

        tokens = re.findall(
            r"[a-z]+|[0-9]+|[^\w\s]",
            text
        )

        return tokens

    def build_vocab(self, texts):

        for text in texts:

            tokens = self.tokenize(text)

            for token in tokens:

                if token not in self.vocab:
                    self.vocab[token] = self.next_id
                    self.next_id += 1

    def encode(
        self,
        text,
        max_length=None,
    ):

        tokens = self.tokenize(text)

        token_ids = [
            self.vocab.get(
                token,
                self.vocab["[UNK]"]
            )
            for token in tokens
        ]

        token_ids = (
            [self.vocab["[CLS]"]]
            + token_ids
            + [self.vocab["[SEP]"]]
        )

        if max_length is not None:

            token_ids = token_ids[:max_length]

            while len(token_ids) < max_length:
                token_ids.append(
                    self.vocab["[PAD]"]
                )

        return token_ids

    @property
    def vocab_size(self):
        return len(self.vocab)