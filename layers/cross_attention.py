import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()

        # cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            batch_first=True
        )

        # optional layernorm
        self.norm = nn.LayerNorm(d)

    def forward(self, z, TFAD_embedding):
        """
        z: [B*patch_num, n_vars, d]
        TFAD_embedding: [B*patch_num, 4, d] 
        """
        attn_out, _ = self.cross_attn(
            query=z,   # [B*patch_num, n_vars, d]
            key=TFAD_embedding,     # [B*patch_num, 4, d]
            value=TFAD_embedding
        )

        # Step4: residual fusion
        z_fused = z + attn_out

        # Step5: layernorm
        z_fused = self.norm(z_fused)

        return z_fused # [B*patch_num, n_vars, d]
