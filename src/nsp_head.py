import torch.nn as nn


class NextSentencePredictionHead(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.classifier = nn.Linear(
            hidden_size,
            2
        )

    def forward(self, pooled_output):

        return self.classifier(
            pooled_output
        )