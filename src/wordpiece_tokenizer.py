class WordPieceTokenizer:

    def __init__(self, vocab):
        self.vocab = vocab

        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"

    def tokenize_word(self, word):

        if word in self.vocab:
            return [word]

        sub_tokens = []
        start = 0

        while start < len(word):

            end = len(word)
            current_subtoken = None

            while start < end:

                substring = word[start:end]

                if start > 0:
                    substring = "##" + substring

                if substring in self.vocab:
                    current_subtoken = substring
                    break

                end -= 1

            if current_subtoken is None:
                return [self.unk_token]

            sub_tokens.append(current_subtoken)
            start = end

        return sub_tokens

    def tokenize(self, words):

        tokens = []

        for word in words:
            tokens.extend(self.tokenize_word(word))

        return tokens