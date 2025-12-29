import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.GMAT.module  models.g'm.module import Attention, PreNorm, FeedForwar
import numpy as np
from utils.config_files_utils import get_params_values
from KNN.efficient_kan.kan import KANLinear
from models.TSViT.simple_Mamba import MambaBlock
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., use_bn=False, use_dyt=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = DyT(dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, MambaBlock(dim, ssm_hidden_dim=dim, conv_kernel=3, dropout=dropout), use_bn=use_bn, use_dyt=use_dyt),
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout), use_bn=use_bn, use_dyt=use_dyt),

                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout), use_bn=use_bn, use_dyt=use_dyt)
            ]))


    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class GMAT(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)

        # self.patch_bn = nn.BatchNorm1d(num_features=self.dim)
        # self.patch_bn = nn.BatchNorm1d(self.dim)

        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))

        #
        self.temporal_transformer = EnhancedAdaptiveFusionTransformer(
            dim=self.dim,
            depth=self.temporal_depth,
            heads=self.heads,
            dim_head=self.dim_head,
            d_state=16,
            expand=2
        )

        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))


        self.space_transformer = EnhancedAdaptiveFusionTransformer(
            dim=self.dim,
            depth=self.spatial_depth,
            heads=self.heads,
            dim_head=self.dim_head,
            d_state=16,
            expand=2
        )



        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(self.dim),
        #     KANLinear(self.dim, self.patch_size ** 2)
        # )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)

        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)

        # B_ = x.shape[0]
        # T_ = x.shape[1]
        # D_ = x.shape[2]
        # x = x.reshape(-1, D_)  # [B_*T_, D_]
        # x = self.patch_bn(x)
        # x = x.reshape(B_, T_, D_)

        # print("After patch embedding:", x.shape)

        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x

    class GMAT(nn.Module):

        def __init__(self, model_config):
            super().__init__()
            self.image_size = model_config['img_res']
            self.patch_size = model_config['patch_size']
            self.num_patches_1d = self.image_size // self.patch_size
            self.num_classes = model_config['num_classes']
            self.num_frames = model_config['max_seq_len']
            self.dim = model_config['dim']
            if 'temporal_depth' in model_config:
                self.temporal_depth = model_config['temporal_depth']
            else:
                self.temporal_depth = model_config['depth']
            if 'spatial_depth' in model_config:
                self.spatial_depth = model_config['spatial_depth']
            else:
                self.spatial_depth = model_config['depth']
            self.heads = model_config['heads']
            self.dim_head = model_config['dim_head']
            self.dropout = model_config['dropout']
            self.emb_dropout = model_config['emb_dropout']
            self.pool = model_config['pool']
            self.scale_dim = model_config['scale_dim']
            assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
            num_patches = self.num_patches_1d ** 2
            patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
                nn.Linear(patch_dim, self.dim), )

            # self.patch_bn = nn.BatchNorm1d(num_features=self.dim)
            # self.patch_bn = nn.BatchNorm1d(self.dim)

            self.to_temporal_embedding_input = nn.Linear(366, self.dim)
            self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))

            # self.temporal_transformer = Transformer(
            #     self.dim, self.temporal_depth, self.heads, self.dim_head,
            #     self.dim * self.scale_dim, self.dropout, use_bn=False, use_dyt=False
            # )
            #
            self.temporal_transformer = EnhancedAdaptiveFusionTransformer(
                dim=self.dim,
                depth=self.temporal_depth,
                heads=self.heads,
                dim_head=self.dim_head,
                d_state=16,
                expand=2
            )

            self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))


            # self.space_transformer = Transformer(
            #     self.dim, self.spatial_depth, self.heads, self.dim_head,
            #     self.dim * self.scale_dim, self.dropout, use_bn=False, use_dyt=False
            # )

            self.space_transformer = EnhancedAdaptiveFusionTransformer(
                dim=self.dim,
                depth=self.spatial_depth,
                heads=self.heads,
                dim_head=self.dim_head,
                d_state=16,
                expand=2
            )

            self.dropout = nn.Dropout(self.emb_dropout)
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.patch_size ** 2)
            )
            # self.mlp_head = nn.Sequential(
            #     nn.LayerNorm(self.dim),
            #     KANLinear(self.dim, self.patch_size ** 2)
            # )

        def forward(self, x):
            x = x.permute(0, 1, 4, 2, 3)
            B, T, C, H, W = x.shape

            xt = x[:, :, -1, 0, 0]
            x = x[:, :, :-1]
            xt = (xt * 365.0001).to(torch.int64)
            xt = F.one_hot(xt, num_classes=366).to(torch.float32)

            xt = xt.reshape(-1, 366)
            temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
            x = self.to_patch_embedding(x)

            # B_ = x.shape[0]
            # T_ = x.shape[1]
            # D_ = x.shape[2]
            # x = x.reshape(-1, D_)  # [B_*T_, D_]
            # x = self.patch_bn(x)
            # x = x.reshape(B_, T_, D_)

            # print("After patch embedding:", x.shape)

            x = x.reshape(B, -1, T, self.dim)
            x += temporal_pos_embedding.unsqueeze(1)
            x = x.reshape(-1, T, self.dim)
            cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
            x = torch.cat((cls_temporal_tokens, x), dim=1)
            x = self.temporal_transformer(x)
            x = x[:, :self.num_classes]
            x = x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
                B * self.num_classes, self.num_patches_1d ** 2, self.dim)
            x += self.space_pos_embedding  # [:, :, :(n + 1)]
            x = self.dropout(x)
            x = self.space_transformer(x)
            x = self.mlp_head(x.reshape(-1, self.dim))
            x = x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2).permute(0, 2, 3, 1)
            x = x.reshape(B, H, W, self.num_classes)
            x = x.permute(0, 3, 1, 2)
            return x

    def update_inference_temporal_position_embeddings(self):
        train_dates_idx = torch.arange(self.train_dates.shape[0])
        min_val = torch.min(self.train_dates).item()
        min_idx = torch.argmin(self.train_dates).item()
        if min_val == 0:
            min_val = torch.min(self.train_dates[1:]).item()
            min_idx += 1
        max_val = torch.max(self.train_dates).item()
        max_idx = torch.argmax(self.train_dates).item()
        pos_eval = torch.zeros(self.eval_dates.shape[0], self.dim)
        for i, evdate in enumerate(self.eval_dates):
            if evdate < min_val:
                pos_eval[i] = self.temporal_pos_embedding[min_idx]
            elif evdate > max_val:
                pos_eval[i] = self.temporal_pos_embedding[max_idx]
            else:
                dist = evdate - self.train_dates
                if 0 in dist:
                    pos_eval[i] = self.temporal_pos_embedding[dist == 0]
                    continue
                lower_idx = train_dates_idx[dist >= 0].max().item()
                upper_idx = train_dates_idx[dist <= 0].min().item()
                lower_date = self.train_dates[lower_idx].item()
                upper_date = self.train_dates[upper_idx].item()
                pos_eval[i] = (upper_date - evdate) / (upper_date - lower_date) * self.temporal_pos_embedding[
                    lower_idx] + \
                              (evdate - lower_date) / (upper_date - lower_date) * self.temporal_pos_embedding[upper_idx]
        self.inference_temporal_pos_embedding = nn.Parameter(pos_eval, requires_grad=False)

    def get_temporal_position_embeddings(self, x):
        B, T = x.shape
        index = torch.bucketize(x.ravel(), self.train_dates)
        return self.temporal_pos_embedding[index].reshape((B, T, self.dim))

    def get_inference_temporal_position_embeddings(self, x):
        B, T = x.shape
        index = torch.bucketize(x.ravel(), self.eval_dates)
        return self.inference_temporal_pos_embedding[index].reshape((B, T, self.dim))



if __name__ == "__main__":
    res = 24
    model_config = {'img_res': res, 'patch_size': 3, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': 20,
                    'max_seq_len': 16, 'dim': 128, 'temporal_depth': 6, 'spatial_depth': 2,
                    'heads': 4, 'pool': 'cls', 'num_channels': 14, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4, 'depth': 4}
    train_config = {'dataset': "psetae_repl_2018_100_3", 'label_map': "labels_20k2k", 'max_seq_len': 16, 'batch_size': 5,
                    'extra_data': [], 'num_workers': 4}

    x = torch.rand((3, 16, res, res, 14))





