import random

import torch
from torch.utils.data import Dataset

from .masking import mask_tokens


class BertPretrainingDataset(Dataset):

    def __init__(
        self,
        texts,
        tokenizer,
        max_length=16,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.mask_token_id = tokenizer.vocab["[MASK]"]
        self.pad_token_id = tokenizer.vocab["[PAD]"]

        self.cls_token_id = tokenizer.vocab["[CLS]"]
        self.sep_token_id = tokenizer.vocab["[SEP]"]

        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        sentence_a = self.texts[index]

        # 50% genuine next sentence
        is_next = random.randint(0, 1)

        if is_next:

            next_index = (
                index + 1
            ) % len(self.texts)

            sentence_b = self.texts[next_index]

            # 0 = IsNext
            nsp_label = 0

        else:

            possible_indices = list(
                range(len(self.texts))
            )

            possible_indices.remove(index)

            random_index = random.choice(
                possible_indices
            )

            sentence_b = self.texts[random_index]

            # 1 = NotNext
            nsp_label = 1

        tokens_a = self.tokenizer.tokenize(
            sentence_a
        )

        tokens_b = self.tokenizer.tokenize(
            sentence_b
        )

        # Reserve:
        # [CLS] + [SEP] + [SEP]
        available_length = (
            self.max_length - 3
        )

        while (
            len(tokens_a) + len(tokens_b)
            > available_length
        ):

            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

        tokens = (
            ["[CLS]"]
            + tokens_a
            + ["[SEP]"]
            + tokens_b
            + ["[SEP]"]
        )

        token_type_ids = (
            [0] * (len(tokens_a) + 2)
            + [1] * (len(tokens_b) + 1)
        )

        input_ids = [
            self.tokenizer.vocab.get(
                token,
                self.tokenizer.vocab["[UNK]"]
            )
            for token in tokens
        ]

        padding_length = (
            self.max_length - len(input_ids)
        )

        input_ids += (
            [self.pad_token_id]
            * padding_length
        )

        token_type_ids += (
            [0] * padding_length
        )

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long
        )

        token_type_ids = torch.tensor(
            token_type_ids,
            dtype=torch.long
        )

        attention_mask = (
            input_ids != self.pad_token_id
        ).long()

        # BERT's 15% dynamic masking
        masked_input_ids, mlm_labels = (
            mask_tokens(
                input_ids,
                mask_token_id=self.mask_token_id,
                vocab_size=self.vocab_size,
                special_token_ids=[
                    self.pad_token_id,
                    self.cls_token_id,
                    self.sep_token_id,
                ],
                probability=0.15,
            )
        )

        return {
            "input_ids": masked_input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "nsp_labels": torch.tensor(
                nsp_label,
                dtype=torch.long
            ),
        }
    