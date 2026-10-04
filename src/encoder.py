import torch.nn as nn

from .attention import MultiHeadSelfAttention


class BertEncoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size=128,
        num_heads=4,
        intermediate_size=512,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.attention_norm = nn.LayerNorm(
            hidden_size
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                hidden_size,
                intermediate_size
            ),
            nn.GELU(),
            nn.Linear(
                intermediate_size,
                hidden_size
            ),
        )

        self.ffn_norm = nn.LayerNorm(
            hidden_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        attention_mask=None,
    ):

        attention_output, attention_weights = (
            self.attention(
                x,
                attention_mask
            )
        )

        x = self.attention_norm(
            x + self.dropout(attention_output)
        )

        ffn_output = self.feed_forward(x)

        x = self.ffn_norm(
            x + self.dropout(ffn_output)
        )

        return x, attention_weights


class BertEncoder(nn.Module):

    def __init__(
        self,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        intermediate_size=512,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            BertEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x,
        attention_mask=None,
    ):

        all_attention_weights = []

        for layer in self.layers:

            x, attention_weights = layer(
                x,
                attention_mask
            )

            all_attention_weights.append(
                attention_weights
            )

        return x, all_attention_weights