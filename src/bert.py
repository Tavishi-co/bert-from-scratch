import torch
import torch.nn as nn

from .embeddings import BertEmbeddings
from .encoder import BertEncoder
from .mlm_head import MaskedLanguageModelHead


class MiniBERT(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        intermediate_size=512,
        max_position_embeddings=128,
        dropout=0.1,
    ):
        super().__init__()

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )

        self.encoder = BertEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

        # BERT's [CLS] representation.
        self.pooler = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.pooler_activation = nn.Tanh()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):

        x = self.embeddings(
            input_ids,
            token_type_ids
        )

        hidden_states, attention_weights = (
            self.encoder(
                x,
                attention_mask
            )
        )

        cls_representation = hidden_states[:, 0]

        pooled_output = self.pooler(
            cls_representation
        )

        pooled_output = self.pooler_activation(
            pooled_output
        )

        return (
            hidden_states,
            pooled_output,
            attention_weights
        )


class BertForMaskedLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        intermediate_size=512,
        max_position_embeddings=128,
        dropout=0.1,
    ):
        super().__init__()

        self.bert = MiniBERT(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )

        self.mlm_head = MaskedLanguageModelHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):

        hidden_states, pooled_output, attention_weights = (
            self.bert(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        )

        logits = self.mlm_head(
            hidden_states
        )

        loss = None

        if labels is not None:

            loss = nn.functional.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
            "pooled_output": pooled_output,
            "attention_weights": attention_weights,
        }