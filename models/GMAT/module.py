from torch import einsum
from einops.layers.torch import Rearrange

from einops import rearrange
from mamba_vision import MambaVisionMixer ,SimpleMambaSSM
import torch
import torch.nn as nn


class EnhancedAdaptiveFusionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=None, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.heads = heads

        # 增强归一化层
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_mamba = nn.LayerNorm(dim)
        self.norm_input = nn.LayerNorm(dim)
        self.norm_gate = nn.LayerNorm(dim)

        # 分支模块 - 添加残差连接
        self.attn = nn.Sequential(
            Attention(dim=dim, heads=heads, dim_head=dim_head),
            nn.LayerNorm(dim)
        )
        self.mamba = nn.Sequential(
            MambaVisionMixer(d_model=dim, d_state=d_state, expand=expand),
            nn.LayerNorm(dim)
        )

        # 动态门控生成器 - 更强大的网络
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # 自适应融合MLP - 增强表达能力
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.GELU()
        )

        # 深度监督输出
        self.aux_output = nn.Linear(dim, dim)

        # 自适应残差
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_norm = self.norm_input(x)

        # 分支处理
        attn_out = self.attn(x_norm)
        mamba_out = self.mamba(x_norm)

        # 动态门控生成
        gate_input = x_norm.mean(dim=1, keepdim=True)
        gate_weights = self.gate_net(self.norm_gate(gate_input))  # [B, 1, dim]
        attn_weights = gate_weights.squeeze(1).detach().cpu().numpy()         # [B, dim]
        mamba_weights = (1.0 - gate_weights).squeeze(1).detach().cpu().numpy()

        # === CSV 保存 ===
        # save_path = r"C:\Users\Think\Desktop\DeepSatModels-main\models\saved_models\PASTIS24\gate_weights_combined.csv"
        # file_exists = os.path.exists(save_path)
        # with open(save_path, mode='a', newline='') as f:
        #     writer = csv.writer(f)
        #     if not file_exists:
        #         header = []
        #         for i in range(attn_weights.shape[1]):
        #             header += [f'attn_channel_{i}', f'mamba_channel_{i}']
        #         writer.writerow(header)
        #     for attn_row, mamba_row in zip(attn_weights, mamba_weights):
        #         writer.writerow([val for pair in zip(attn_row, mamba_row) for val in pair])

        # 门控融合
        fused = gate_weights * attn_out + (1 - gate_weights) * mamba_out
        fused = self.fusion_mlp(fused)

        out = self.output_norm(x + self.residual_scale * fused)
        aux_out = self.aux_output(out)

        return out, aux_out



class EnhancedAdaptiveFusionTransformer(nn.Module):
    def __init__(self, dim, depth=4, heads=4, dim_head=None, d_state=16, expand=2):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([
            EnhancedAdaptiveFusionBlock(dim, heads, dim_head, d_state, expand)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # 深度监督权重
        self.aux_weights = nn.Parameter(torch.ones(depth))

    def forward(self, x):
        aux_outputs = []
        for i, layer in enumerate(self.layers):
            x, aux_out = layer(x)
            aux_outputs.append(aux_out)

        # 加权聚合深度监督输出
        weights = torch.softmax(self.aux_weights, dim=0)
        weighted_aux = sum(w * out for w, out in zip(weights, aux_outputs))

        # 主输出
        main_output = self.norm(x)

        return main_output + weighted_aux



class OptimizedAdaptiveFusionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=None, d_state=16, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Attention分支
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)

        # Mamba分支
        self.mamba = MambaVisionMixer(d_model=dim, d_state=d_state, expand=expand)

        # 门控融合机制（更强的非线性映射）
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 将两个分支的输出拼接后映射到dim
            nn.GELU(),  # 激活
            nn.Linear(dim, 2),  # 变为2个权重，分别对应每个分支的加权
            nn.Softmax(dim=-1)  # 使用Softmax来得到每个分支的权重
        )

        # 自适应残差缩放因子（可调）
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # 双分支处理：Attention和Mamba
        attn_out = self.attn(x)
        mamba_out = self.mamba(x)

        # 门控融合：将两个分支的输出拼接后计算权重
        combined = torch.cat([attn_out, mamba_out], dim=-1)  # 拼接两个输出
        gate_weights = self.gate(combined)  # 计算门控权重，输出两个权重
        fused = gate_weights[:, :, 0:1] * attn_out + gate_weights[:, :, 1:2] * mamba_out  # 根据权重融合

        # 自适应残差连接：将融合结果加回原输入，经过res_scale缩放
        return residual + self.res_scale * fused


class AdaptiveFusionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, d_state=16, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Attention分支
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)

        # Mamba分支
        self.mamba = MambaVisionMixer(d_model=dim, d_state=d_state, expand=expand)

        # 门控融合机制（Gated Fusion）
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 将两个分支的输出拼接后映射到dim
            nn.GELU(),  # 非线性激活
            nn.Linear(dim, 2),  # 输出2个权重，Softmax处理
            nn.Softmax(dim=-1)  # 归一化权重
        )

        # 自适应残差缩放因子
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x  # 保存输入x，供残差连接使用
        x = self.norm(x)  # 归一化

        # 双分支处理：Attention和Mamba
        attn_out = self.attn(x)  # Attention分支的输出
        mamba_out = self.mamba(x)  # Mamba SSM分支的输出

        # 门控融合：将两个分支的输出拼接后计算权重
        combined = torch.cat([attn_out, mamba_out], dim=-1)  # 拼接两个输出
        gate_weights = self.gate(combined)  # 计算门控权重，输出2个权重值
        fused = gate_weights[:, :, 0:1] * attn_out + gate_weights[:, :, 1:2] * mamba_out  # 根据权重融合

        # 自适应残差连接：将融合结果加回原输入，经过res_scale缩放
        return residual + self.res_scale * fused  # 残差连接


class ParallelBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, d_state=16, expand=2, mlp_expand=2, dropout=0.):
        super().__init__()
        dim_head = dim // heads if dim_head is None else dim_head
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ssm = SimpleMambaSSM(d_model=dim, d_state=d_state, expand=expand)
        self.fuse = nn.Linear(dim * 2, dim)   # 融合Attention和SSM输出（拼接后映射回原维度）
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_expand),
            nn.GELU(),
            nn.Linear(dim * mlp_expand, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x1 = self.attn(self.norm1(x))        # (B, N, D)
        x2 = self.ssm(self.norm1(x))         # (B, N, D)
        x_cat = torch.cat([x1, x2], dim=-1)  # (B, N, 2D)
        x_fused = self.fuse(x_cat)           # (B, N, D)
        x = x + x_fused                      # 残差
        x = x + self.mlp(self.norm2(x))      # 残差
        return x



class ParallelTransformer(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_head=None, d_state=16, expand=2, mlp_expand=2, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            ParallelBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                d_state=d_state,
                expand=expand,
                mlp_expand=mlp_expand,
                dropout=dropout
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.norm(x)




class MixedBlock(nn.Module):
    def __init__(self, dim, use_attention=True, d_state=16, expand=2, mlp_expand=2, dropout=0.):
        super().__init__()
        self.use_attention = use_attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if use_attention:
            self.mixer = Attention(dim=dim, heads=8, dim_head=dim//8, dropout=dropout)
        else:
            self.mixer = SimpleMambaSSM(d_model=dim, d_state=d_state, expand=expand)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_expand),
            nn.GELU(),
            nn.Linear(dim*mlp_expand, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MixedTransformer(nn.Module):
    def __init__(self, dim, depth, attn_idx=None, d_state=16, expand=2, mlp_expand=2, dropout=0.):
        super().__init__()
        if attn_idx is None:
            # 默认前半Attention，后半SSM
            attn_idx = set(range(depth//2))
        else:
            attn_idx = set(attn_idx)
        self.layers = nn.ModuleList([
            MixedBlock(dim, use_attention=(i in attn_idx), d_state=d_state, expand=expand, mlp_expand=mlp_expand, dropout=dropout)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.norm(x)




class MambaBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = MambaVisionMixer(d_model=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



class SSMTransformer(nn.Module):
    def __init__(self, dim, depth, d_state=16, expand=2, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                SimpleMambaSSM(dim, d_state=d_state, expand=expand),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim*expand),
                    nn.GELU(),
                    nn.Linear(dim*expand, dim),
                    nn.Dropout(dropout)
                )
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for layer in self.layers:
            x = x + layer[1](layer[0](x))  # SSM残差
            x = x + layer[3](layer[2](x))  # MLP残差
        return self.norm(x)


class MambaTransformer(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(dim, mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.norm(x)




class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, use_bn=False, use_dyt=False):
        super().__init__()
        self.use_dyt = use_dyt
        if use_dyt:
            self.norm = DyT(dim)  # 这里的dim即特征维度
        elif use_bn:
            self.norm = nn.BatchNorm1d(dim)
        else:
            self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.use_dyt:
            # DyT 直接作用在 [B, N, D] 形状的输入上
            return self.fn(self.norm(x), **kwargs)
        elif hasattr(self, "use_bn") and self.use_bn:
            B, seq, dim = x.shape
            x_ = x.contiguous().view(-1, dim)
            x_ = self.norm(x_)
            x = x_.view(B, seq, dim)
            return self.fn(x, **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)




class PreNormLocal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        # print('before fn: ', x.shape)
        x = self.fn(x, **kwargs)
        # print('after fn: ', x.shape)
        return x


class Conv1x1Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print(q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                     )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=14, w=14)
            )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
