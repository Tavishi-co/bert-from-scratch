import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings=512,
        type_vocab_size=2,
        dropout=0.1,
    ):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            vocab_size,
            hidden_size
        )

        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size
        )

        self.token_type_embeddings = nn.Embedding(
            type_vocab_size,
            hidden_size
        )

        self.layer_norm = nn.LayerNorm(
            hidden_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
    ):
        batch_size, seq_len = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(
                input_ids
            )

        position_ids = torch.arange(
            seq_len,
            device=input_ids.device
        )

        position_ids = position_ids.unsqueeze(
            0
        ).expand(
            batch_size,
            seq_len
        )

        token_embeddings = self.word_embeddings(
            input_ids
        )

        position_embeddings = self.position_embeddings(
            position_ids
        )

        segment_embeddings = self.token_type_embeddings(
            token_type_ids
        )

        embeddings = (
            token_embeddings
            + position_embeddings
            + segment_embeddings
        )

        embeddings = self.layer_norm(
            embeddings
        )

        embeddings = self.dropout(
            embeddings
        )

        return embeddings