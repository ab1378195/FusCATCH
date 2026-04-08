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


class BiCrossAttentionGatedFusion(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.attn_z_to_t = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_t_to_z = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_z = nn.LayerNorm(d)
        self.norm_t = nn.LayerNorm(d)
        self.gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    def forward(self, z, TFAD_embedding):
        if z.dim() != 3 or TFAD_embedding.dim() != 3:
            raise ValueError(f"Expected 3D tensors, got z={z.shape}, TFAD_embedding={TFAD_embedding.shape}")
        if z.shape[0] != TFAD_embedding.shape[0] or z.shape[-1] != TFAD_embedding.shape[-1]:
            raise ValueError(f"Shape mismatch: z={z.shape}, TFAD_embedding={TFAD_embedding.shape}")

        tfad_ctx, _ = self.attn_z_to_t(
            query=TFAD_embedding,
            key=z,
            value=z,
        )
        tfad_ref = self.norm_t(TFAD_embedding + tfad_ctx)

        z_ctx, _ = self.attn_t_to_z(
            query=z,
            key=tfad_ref,
            value=tfad_ref,
        )
        gate = torch.sigmoid(self.gate(torch.cat([z, z_ctx], dim=-1)))
        z_out = self.norm_z(z + gate * z_ctx)
        return z_out
