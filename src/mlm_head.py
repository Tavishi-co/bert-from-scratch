import torch
import torch.nn as nn


class MaskedLanguageModelHead(nn.Module):

    def __init__(
        self,
        hidden_size,
        vocab_size,
    ):
        super().__init__()

        self.transform = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        self.decoder = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False
        )

        self.bias = nn.Parameter(
            torch.zeros(vocab_size)
        )

    def forward(self, hidden_states):

        x = self.transform(
            hidden_states
        )

        logits = self.decoder(x)

        logits = logits + self.bias

        return logits