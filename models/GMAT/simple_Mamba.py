import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMamba(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, kernel_size=3):
        super().__init__()
        self.expand = expand
        self.d_inner = d_model * expand
        self.d_model = d_model

        self.input_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size, padding=kernel_size-1, groups=self.d_inner)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x: (B, L, D)
        xz = self.input_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)  # to (B, D, L)
        x = self.conv(x)[..., :x.size(-1)]  # causal conv
        x = self.act(x)
        x = x.transpose(1, 2)  # back to (B, L, D)
        x = self.x_proj(x)
        x = x * self.act(z)
        return self.out_proj(x)

# MambaBlock: 堆叠多个 SimpleMamba
class MambaBlock(nn.Module):
    def __init__(self, d_model, depth=4):
        super().__init__()
        self.layers = nn.Sequential(*[SimpleMamba(d_model) for _ in range(depth)])

    def forward(self, x):
        return self.layers(x)




