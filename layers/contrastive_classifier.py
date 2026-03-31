import torch
from torch import nn


class ContrastiveClasifier(nn.Module):
    def __init__(self, distance: nn.Module):
        super().__init__()
        self.distance = distance
        self.eps = 1e-10

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> torch.Tensor:
        xt = torch.cat((x1, x3), -1)
        yt = torch.cat((x2, x4), -1)
        dists = self.distance(xt,yt)
        # Probability of the two embeddings being equal: exp(-dist)
        log_prob_equal = -dists
        # Computation of log_prob_different
        prob_different = torch.clamp(1 - torch.exp(log_prob_equal), self.eps, 1)
        log_prob_different = torch.log(prob_different)
        logits_different = log_prob_different - log_prob_equal
        return logits_different