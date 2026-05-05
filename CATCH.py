import torch
import torch.nn as nn
from layers.RevIn import RevIN
from layers.mask import channel_mask_generator
from layers.transformer import Trans_C
from layers.flatten_head import Flatten_Head
from layers.TFAD import TFAD
from layers.cross_attention import BiCrossAttentionGatedFusion

class CATCHModel(nn.Module):
    def __init__(self, configs):
        super(CATCHModel, self).__init__()
        self.revin_layer = RevIN(
            configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last
        )
        # Patching
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.horizon = self.seq_len
        patch_num = int(
            (configs.seq_len - configs.patch_size) / configs.patch_stride + 1
        )
        self.norm = nn.LayerNorm(self.patch_size)
        # Backbone
        self.re_attn = True
        self.mask_generator = channel_mask_generator(
            input_size=configs.patch_size, n_vars=configs.c_in
        )
        self.frequency_transformer = Trans_C(
            dim=configs.cf_dim,
            depth=configs.e_layers,
            heads=configs.n_heads,
            mlp_dim=configs.d_ff,
            dim_head=configs.head_dim,
            dropout=configs.dropout,
            patch_dim=configs.patch_size * 2,
            horizon=self.horizon * 2,
            d_model=configs.d_model * 2,
            regular_lambda=configs.regular_lambda,
            temperature=configs.temperature,
        )
        # Head
        self.head_nf_f = configs.d_model * 2 * patch_num
        self.n_vars = configs.c_in
        self.individual = configs.individual
        self.head_f1 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            configs.seq_len,
            head_dropout=configs.head_dropout,
        )
        self.head_f2 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            configs.seq_len,
            head_dropout=configs.head_dropout,
        )

        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.rfftlayer = nn.Linear(self.seq_len * 2 - 2, self.seq_len)
        self.final = nn.Linear(self.seq_len * 2, self.seq_len)

        # break up R&I:
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        # TFAD
        self.TFAD = TFAD(configs)
        self.cross_attention = BiCrossAttentionGatedFusion(configs.d_model * 2, configs.num_heads)

    def forward(self, z):  # z: [bs x seq_len x n_vars]
        z = self.revin_layer(z, 'norm') 
        z = z.permute(0, 2, 1) # [bs x n_vars x seq_len]

        # TFAD module
        TFAD_outputs = self.TFAD(z)
        TFAD_score = TFAD_outputs["score"]
        TFAD_embedding = TFAD_outputs["embedding"]
        TFAD_alpha = TFAD_outputs["alpha"]
        
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        # do patching
        z1 = z1.unfold(
            dimension=-1, size=self.patch_size, step=self.patch_stride
        )  # z1: [bs x nvars x patch_num x patch_size]
        z2 = z2.unfold(
            dimension=-1, size=self.patch_size, step=self.patch_stride
        )  # z2: [bs x nvars x patch_num x patch_size]

        # for channel-wise_1
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        # model shape
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_size = z1.shape[3]

        # proposed
        z1 = torch.reshape(
            z1, (batch_size * patch_num, c_in, z1.shape[-1])
        )  # z: [bs * patch_num,nvars, patch_size]
        z2 = torch.reshape(
            z2, (batch_size * patch_num, c_in, z2.shape[-1])
        )  # z: [bs * patch_num,nvars, patch_size]
        z_cat = torch.cat((z1, z2), -1)

        channel_mask = self.mask_generator(z_cat)

        z, dcloss = self.frequency_transformer(z_cat, channel_mask) # z: [bs * patch_num,nvars, d_model*2]
        z_freq = z

        TFAD_embedding = TFAD_embedding.unsqueeze(1).repeat(1, patch_num, 1, 1)
        TFAD_embedding = TFAD_embedding.reshape(batch_size * patch_num, 4, -1)
        tfad_embedding_patch = TFAD_embedding
        z_fused = self.cross_attention(z, TFAD_embedding)
        

        z1 = self.get_r(z_fused)
        z2 = self.get_i(z_fused)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)  # z1: [bs, nvars， patch_num, horizon]
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)  # z: [bs x nvars x seq_len]
        z2 = self.head_f2(z2)  # z: [bs x nvars x seq_len]

        complex_z = torch.complex(z1, z2)

        z = torch.fft.ifft(complex_z)
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        # denorm
        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')
        # return {"z":z, "complex_z":complex_z.permute(0, 2, 1), "dcloss":dcloss, "TFAD_score":TFAD_score, "z_freq":z_freq, "tfad_embedding_patch":tfad_embedding_patch}
        return {"z":z, "complex_z":complex_z.permute(0, 2, 1), "dcloss":dcloss, "TFAD_score":TFAD_score, "z_freq":z_freq, "tfad_embedding_patch":tfad_embedding_patch, "TFAD_alpha": TFAD_alpha}
