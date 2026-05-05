import torch.nn as nn
import torch
from layers.distances import CosineDistance, LpDistance, BinaryOnX1
from layers.contrastive_classifier import ContrastiveClasifier
from layers.tcn_encoder import TCNEncoder

def D_matrix(N):
    D = torch.zeros(N - 1, N)
    D[:, 1:] = torch.eye(N - 1)
    D[:, :-1] -= torch.eye(N - 1)
    return D

class hp_filter(nn.Module):
    """
        Hodrick Prescott Filter to decompose the series
    """

    def __init__(self, lamb, seq_len):
        super(hp_filter, self).__init__()
        self.lamb = lamb
        self.N = seq_len
        D1 = D_matrix(self.N)
        D2 = D_matrix(self.N-1)
        D = torch.mm(D2, D1)
        self.register_buffer("filter_matrix", torch.inverse(torch.eye(self.N) + self.lamb * torch.mm(D.T, D)))

    def forward(self, x):
        # x: [batch_size, n_vars, seq_len]
        x = x.permute(0, 2, 1) # [batch_size, seq_len, n_vars]
        if x.shape[1] == self.N:
            g = torch.matmul(self.filter_matrix, x)
        else:
            # 可能冗余
            print("activate")
            N = x.shape[1]
            D1 = D_matrix(N)
            D2 = D_matrix(N-1)
            D = torch.mm(D2, D1).to(x.device)
            g = torch.matmul(torch.inverse(torch.eye(N).to(x.device) + self.lamb * torch.mm(D.T, D)), x)
        res = x - g
        g = g.permute(0, 2, 1)
        res = res.permute(0, 2, 1)
        return res, g

class TFAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.suspect_window_length = config.suspect_window_length
        # Decomposition Network
        self.Decomp = hp_filter(lamb=config.hp_lamb, seq_len=config.seq_len)
        # Encoder Network
        self.encoder1 = TCNEncoder(
            in_channels=config.c_in,
            out_channels=config.d_model * 2,
            kernel_size=config.tcn_kernel_size,
            tcn_channels=config.tcn_out_channels,
            tcn_layers=config.tcn_layers,
            tcn_out_channels=config.tcn_out_channels,
            maxpool_out_channels=config.tcn_maxpool_out_channels,
            normalize_embedding=config.normalize_embedding,
        )
        
        self.encoder2 = TCNEncoder(
            in_channels=config.c_in,
            out_channels=config.d_model * 2,
            kernel_size=config.tcn_kernel_size,
            tcn_channels=config.tcn_out_channels,
            tcn_layers=config.tcn_layers,
            tcn_out_channels=config.tcn_out_channels,
            maxpool_out_channels=config.tcn_maxpool_out_channels,
            normalize_embedding=config.normalize_embedding,
        )

        # Contrast Classifier
        if config.distance == "cosine":
            # For the contrastive approach, the cosine distance is used
            distance = CosineDistance()
        elif config.distance == "L2":
            # For the contrastive approach, the L2 distance is used
            distance = LpDistance(p=2)
        elif config.distance == "non-contrastive":
            # For the non-contrastive approach, the classifier is
            # a neural-net based on the embedding of the whole window
            distance = BinaryOnX1(rep_dim=config.d_model * 2, layers=1)
        self.classifier = ContrastiveClasifier(
            distance=distance,
        )
        # Feature Weighting (Soft Gating)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, c_in, window_length)
        res, cyc = self.Decomp(x)
        # Encoder Network
        whole_res_emb = self.encoder1(res)
        context_res_emb = self.encoder1(res[..., : -self.suspect_window_length])
        whole_cyc_emb = self.encoder2(cyc)
        context_cyc_emb = self.encoder2(cyc[..., : -self.suspect_window_length])
        # Contrast Classifier
        logits_anomaly = self.classifier(whole_res_emb, context_res_emb, whole_cyc_emb, context_cyc_emb)        
        
        embedding = torch.stack([whole_res_emb, context_res_emb, whole_cyc_emb, context_cyc_emb], dim=1) # [batch_size, 4, d_model * 2]
       
        score = self.mlp(embedding).squeeze(-1) # [batch_size, 4]
        alpha = torch.softmax(score, dim=1) # [batch_size, 4]
        weighted_embedding = embedding * alpha.unsqueeze(-1) # [batch_size, 4, d_model * 2]
        # return {"score":logits_anomaly, "embedding":weighted_embedding}
        return {"score":logits_anomaly, "embedding":weighted_embedding, "alpha": alpha}

