import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import sys
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
import matplotlib as mpl
from data import get_dataloaders
from data.PASTIS24.data_transforms import Normalize
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
plt.style.use('seaborn-whitegrid')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


CFG_PATH = r"C:\Users\Think\Desktop\DeepSatModels-main\configs\PASTIS24\TSViT_fold5.yaml"
WEIGHTS_PATH = r"C:\Users\Think\Desktop\模型\logs\门控自适应\best.pth"
SAVE_DIR = r"C:\Users\Think\Desktop\attention_analysis"
PICKLE_FILE = r"C:\Users\Think\Desktop\bq\bq_new_new\kuochong_30\64\total2\20369_1_0.pickle"
NUM_SAMPLES = 1
DEVICE_IDS = [0]
ANALYSIS_MODE = 'spatial' 


def save_gate_weights(weights_data, save_dir, prefix, num_positions):

    os.makedirs(save_dir, exist_ok=True)
    if not weights_data:
        print("⚠️ 警告: 权重数据为空，无法保存CSV")
        return

    for block_idx, block_data in enumerate(weights_data):
        if 'attn' not in block_data or 'mamba' not in block_data:
            print(f"⚠️ 块 {block_idx} 缺少权重数据，跳过保存")
            continue

        attn_weights = block_data['attn']
        mamba_weights = block_data['mamba']
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.numpy()
        if isinstance(mamba_weights, torch.Tensor):
            mamba_weights = mamba_weights.numpy()

        if attn_weights.ndim > 1:
            attn_weights = attn_weights[0]
        if mamba_weights.ndim > 1:
            mamba_weights = mamba_weights[0]

        if len(attn_weights) < num_positions:
            padded = np.zeros(num_positions)
            padded[:len(attn_weights)] = attn_weights
            attn_weights = padded
        elif len(attn_weights) > num_positions:
            attn_weights = attn_weights[:num_positions]

        if len(mamba_weights) < num_positions:
            padded = np.zeros(num_positions)
            padded[:len(mamba_weights)] = mamba_weights
            mamba_weights = padded
        elif len(mamba_weights) > num_positions:
            mamba_weights = mamba_weights[:num_positions]

        df = pd.DataFrame({
            'position': range(num_positions),
            'attn_weight': attn_weights,
            'mamba_weight': mamba_weights
        })

        csv_path = os.path.join(save_dir, f"{prefix}_gate_weights_block_{block_idx}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ 保存门控权重CSV: {csv_path}")


def plot_gate_weights(weights_data, save_dir, prefix, num_positions, figsize=(10, 6)):

    os.makedirs(save_dir, exist_ok=True)

    if not weights_data:
        print("⚠️ 警告: 权重数据为空，无法绘制图表")
        return

    for block_idx, block_data in enumerate(weights_data):
        if 'attn' not in block_data or 'mamba' not in block_data:
            print(f"⚠️ 块 {block_idx} 缺少权重数据，跳过绘图")
            continue

        attn_weights = block_data['attn']
        mamba_weights = block_data['mamba']

        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.numpy()
        if isinstance(mamba_weights, torch.Tensor):
            mamba_weights = mamba_weights.numpy()

        if attn_weights.ndim > 1:
            attn_weights = attn_weights[0]
        if mamba_weights.ndim > 1:
            mamba_weights = mamba_weights[0]

        if len(attn_weights) > num_positions:
            attn_weights = attn_weights[:num_positions]
        if len(mamba_weights) > num_positions:
            mamba_weights = mamba_weights[:num_positions]

        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        attention_color = '#4e79a7' 
        mamba_color = '#f28e2b'

        positions = np.arange(num_positions)
        bar_width = 0.4

        ax.bar(
            positions,
            attn_weights,
            width=bar_width,
            color=attention_color,
            edgecolor='black',
            linewidth=0.7,
            alpha=0.9,
            label='Attention'
        )

        ax.bar(
            positions + bar_width,
            mamba_weights,
            width=bar_width,
            color=mamba_color,
            edgecolor='black',
            linewidth=0.7,
            alpha=0.9,
            label='Mamba'
        )

        for i, (attn_w, mamba_w) in enumerate(zip(attn_weights, mamba_weights)):
            ax.text(
                i, attn_w + 0.005, f"{attn_w:.3f}",
                ha='center', fontsize=8, fontweight='bold'
            )
            ax.text(
                i + bar_width, mamba_w + 0.005, f"{mamba_w:.3f}",
                ha='center', fontsize=8, fontweight='bold'
            )

        ax.set_title(f'Gate Weights - Block {block_idx + 1}', fontsize=14, pad=15)
        ax.set_xlabel('Patch Position', fontsize=12, labelpad=10)
        ax.set_ylabel('Gate Weight Value', fontsize=12, labelpad=10)
        ax.set_xticks(positions + bar_width / 2)
        ax.set_xticklabels([f'{i}' for i in range(num_positions)], fontsize=10)

        max_val = max(np.max(attn_weights), np.max(mamba_weights))
        ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1.0)

        ax.legend(fontsize=10, frameon=True, shadow=True, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        img_path = os.path.join(save_dir, f"{prefix}_gate_weights_block_{block_idx + 1}.png")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        print(f"{img_path}")


def custom_normalize(data, mean, std):
    mean = mean.squeeze().astype(np.float32)
    std = std.squeeze().astype(np.float32)

    if data.ndim == 4:  # (T, C, H, W)
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)
    elif data.ndim == 3:  # (C, H, W)
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
    else:
        raise ValueError(f"不支持的输入维度: {data.ndim}")

    normalized = (data - mean) / std
    return normalized.astype(np.float32)


def prepare_model_input(normalized_img, doys):
    doy_normalized = doys / 365.0
    doy_channel = doy_normalized[:, np.newaxis, np.newaxis, np.newaxis]
    doy_channel = np.broadcast_to(
        doy_channel,
        (doy_normalized.shape[0], 1, normalized_img.shape[2], normalized_img.shape[3])
    )

    model_input = np.concatenate([normalized_img, doy_channel], axis=1)
    return model_input.astype(np.float32)


def process_time_features(xt, device):
    xt = torch.clamp(xt * 365.0001, 0, 365)
    xt = xt.to(torch.int64)
    max_val = xt.max().item()
    if max_val >= 366:
        print(f"⚠️ 警告: 最大时间特征值 {max_val} 超过365，将被裁剪")
        xt = torch.clamp(xt, 0, 365)
    xt = F.one_hot(xt, num_classes=366).to(torch.float32)
    return xt


def plot_attention_bipartite(attention_matrix, save_path, block_idx, figsize=(12, 6)):
    T = attention_matrix.shape[0] 

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    key_importance = attention_matrix.sum(axis=0)
    query_importance = attention_matrix.sum(axis=1)

    key_importance_norm = (key_importance - key_importance.min()) / (key_importance.max() - key_importance.min() + 1e-8)
    query_importance_norm = (query_importance - query_importance.min()) / (
            query_importance.max() - query_importance.min() + 1e-8)

    min_node_size = 50
    max_node_size = 300
    key_sizes = min_node_size + (max_node_size - min_node_size) * key_importance_norm
    query_sizes = min_node_size + (max_node_size - min_node_size) * query_importance_norm

    important_key_indices = np.where(key_importance > key_importance.mean())[0]
    important_query_indices = np.where(query_importance > query_importance.mean())[0]

    x_pos_top = np.linspace(0, 1, T) 
    x_pos_bottom = np.linspace(0, 1, T)

    all_key_nodes = ax.scatter(
        x_pos_top, [1] * T,
        s=key_sizes,
        c='lightgray',
        edgecolors='k',
        alpha=0.7,
        label='输入节点 (Key)'
    )

    all_query_nodes = ax.scatter(
        x_pos_bottom, [0] * T,
        s=query_sizes,
        c='lightgray',
        edgecolors='k',
        alpha=0.7,
        label='输出节点 (Query)'
    )

    if len(important_key_indices) > 0:
        ax.scatter(
            x_pos_top[important_key_indices], [1] * len(important_key_indices),
            s=key_sizes[important_key_indices],
            c='skyblue',
            edgecolors='k',
            alpha=1.0,
            zorder=10 
        )

    if len(important_query_indices) > 0:
        ax.scatter(
            x_pos_bottom[important_query_indices], [0] * len(important_query_indices),
            s=query_sizes[important_query_indices],
            c='lightblue',
            edgecolors='k',
            alpha=1.0,
            zorder=10  
        )


    max_weight = np.max(attention_matrix)
    min_weight = np.min(attention_matrix[attention_matrix > 0])

    max_attention_idx = np.unravel_index(np.argmax(attention_matrix), attention_matrix.shape)

    weight_threshold = max_weight * 0.05 

    for i in range(T): 
        for j in range(T):  
            weight = attention_matrix
            if weight > weight_threshold:
                norm_weight = (weight - min_weight) / (max_weight - min_weight + 1e-8)
                linewidth = 0.5 + 2.5 * norm_weight

                is_max = (i, j) == max_attention_idx

                is_important_edge = (j in important_key_indices) and (i in important_query_indices)

                if is_max:
                    color = 'lightblue'
                    alpha = 0.8
                    linestyle = '-'
                elif is_important_edge:
                    color = 'lightblue'
                    alpha = 0.8
                    linestyle = '-'
                else:
                    color = 'lightgray'
                    alpha = 0.3
                    linestyle = '--' 

                ax.plot(
                    [x_pos_top[j], x_pos_bottom[i]],
                    [1, 0],
                    linewidth=linewidth,
                    alpha=alpha,
                    color=color,
                    linestyle=linestyle,
                    zorder=1 if is_important_edge or is_max else 0
                )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{save_path}")
    return save_path


def modified_forward_with_attention(inputs, net, mode='temporal'):

    B, T, C, H, W = inputs.shape
    inputs = inputs.float()

    xt = inputs[:, :, -1, 0, 0]
    xt = process_time_features(xt, inputs.device)
    xt = xt.reshape(-1, 366)

    temporal_pos_embedding = net.to_temporal_embedding_input(xt).reshape(B, T, net.dim)

    x = inputs[:, :, :-1] 

    assert H % net.patch_size == 0, f"高度 {H} 不能被 patch_size {net.patch_size} 整除"
    assert W % net.patch_size == 0, f"宽度 {W} 不能被 patch_size {net.patch_size} 整除"

    num_patches_h = H // net.patch_size
    num_patches_w = W // net.patch_size
    num_patches = num_patches_h * num_patches_w

    x = x.unfold(3, net.patch_size, net.patch_size)
    x = x.unfold(4, net.patch_size, net.patch_size)
    x = x.permute(0, 3, 4, 1, 2, 5, 6)
    x = x.reshape(B * num_patches_h * num_patches_w, T, 20 * net.patch_size * net.patch_size)

    x = net.to_patch_embedding[1](x)

    x = x.reshape(B, num_patches, T, net.dim)
    x += temporal_pos_embedding.unsqueeze(1)
    x = x.reshape(B * num_patches, T, net.dim)

    cls_temporal_tokens = net.temporal_token.repeat(B * num_patches, 1, 1)
    x = torch.cat((cls_temporal_tokens, x), dim=1)

    if mode == 'temporal':
        for block in net.temporal_transformer.layers:
            if hasattr(block, 'attn'):
                block.attn.return_attention = True  
        x = net.temporal_transformer(x)

        attention_scores_list = []
        for block in net.temporal_transformer.layers:
            if hasattr(block, 'attention_scores') and block.attention_scores is not None:
                attention_scores_list.append(block.attention_scores)
    else:
        x = net.temporal_transformer(x)
        attention_scores_list = []

    x = x[:, :net.num_classes]
    x = x.reshape(B, num_patches, net.num_classes, net.dim)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B * net.num_classes, num_patches, net.dim)

    space_pos_embedding = net.space_pos_embedding[:, :num_patches] if net.space_pos_embedding.shape[
                                                                          1] > num_patches else net.space_pos_embedding
    x += space_pos_embedding

    if hasattr(net, 'dropout'):
        x = net.dropout(x)
    if mode == 'spatial':
        for block in net.space_transformer.layers:
            if hasattr(block, 'attn'):
                block.attn.return_attention = True
        x = net.space_transformer(x)

        attention_scores_list = []
        for block in net.space_transformer.layers:
            if hasattr(block, 'attention_scores') and block.attention_scores is not None:
                attention_scores_list.append(block.attention_scores)
    else:
        x = net.space_transformer(x)

    x = net.mlp_head(x.reshape(-1, net.dim))

    x = x.reshape(B, net.num_classes, num_patches, net.patch_size ** 2)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(B, num_patches_h, num_patches_w, net.patch_size, net.patch_size, net.num_classes)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, num_patches_h * net.patch_size, num_patches_w * net.patch_size, net.num_classes)
    x = x.permute(0, 3, 1, 2)
    return x, attention_scores_list


def compute_spatial_block_importance_per_layer(attention_scores_list, num_blocks=16):

    block_importance_per_layer = []

    for layer_idx, attention_scores in enumerate(attention_scores_list):
        print(f"计算第 {layer_idx + 1} 层空间块重要性...")

        if attention_scores.dim() == 4:
            avg_attention = attention_scores.mean(dim=1)[0]  # [query_len, key_len]

            block_importance = avg_attention.sum(dim=0).detach().cpu().numpy()

            if len(block_importance) < num_blocks:
                block_importance = np.pad(block_importance, (0, num_blocks - len(block_importance)),
                                          mode='constant', constant_values=0)
            elif len(block_importance) > num_blocks:
                block_importance = block_importance[:num_blocks]

            block_importance_per_layer.append(block_importance)
        else:
            block_importance_per_layer.append(np.full(num_blocks, np.nan))

    return block_importance_per_layer

def plot_temporal_attention(attention_scores, save_dir, block_idx, input_series=None):

    if attention_scores.dim() == 4:
        avg_attention = attention_scores.mean(dim=1)[0]  # [query_len, key_len]
        print(f"平均注意力形状: {avg_attention.shape}")

        num_timesteps = 21
        timestep_attention = avg_attention[-num_timesteps:, -num_timesteps:]
        print(f"时间步注意力形状: {timestep_attention.shape}")

        timestep_attention = timestep_attention.detach().cpu().numpy()

        csv_path = os.path.join(save_dir, f"timestep_attention_block_{block_idx}.csv")
        pd.DataFrame(timestep_attention).to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(timestep_attention, annot=False, cmap="viridis", cbar=True, square=True,
                         annot_kws={"size": 16})
        plt.xlabel("observation time t_out", fontsize=20)
        plt.ylabel("observation time t_in", fontsize=20)
        ax.tick_params(axis='x', labelsize=16) 
        ax.tick_params(axis='y', labelsize=16)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

        img_path = os.path.join(save_dir, f"timestep_attention_heatmap_block_{block_idx}.png")
        plt.savefig(img_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f" {img_path} (形状: {timestep_attention.shape})")

        bipartite_path = os.path.join(save_dir, f"bipartite_block_{block_idx}.png")
        plot_attention_bipartite(
            timestep_attention,
            bipartite_path,
            block_idx
        )
        return csv_path, img_path

    else:
        print(f" {attention_scores.dim()}")
        return None, None


def plot_spatial_attention(attention_scores, save_dir, block_idx):
    print(f"块 {block_idx} {attention_scores.shape}")

    if attention_scores.dim() == 4:
        avg_attention = attention_scores.mean(dim=1)[0].detach().cpu().numpy()

        csv_path = os.path.join(save_dir, f"spatial_attention_block_{block_idx}.csv")
        pd.DataFrame(avg_attention).to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(avg_attention, annot=False, cmap="Reds", cbar=True, square=True)

        plt.xlabel("Key", fontsize=20)
        plt.ylabel("Query", fontsize=20)

        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

        img_path = os.path.join(save_dir, f"spatial_attention_heatmap_block_{block_idx}.png")
        plt.tight_layout()
        plt.savefig(img_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f" {img_path} (形状: {avg_attention.shape})")

        return csv_path, img_path
    else:
        print(f" {attention_scores.dim()}")
        return None, None

def plot_attention_over_time(attention_scores, save_dir, block_idx):
    num_timesteps = 21 

    class_tokens_count = 4 
    timestep_start_idx = class_tokens_count 

    if attention_scores.dim() == 4:  # [batch, heads, query_len, key_len]
        cls_attention = attention_scores[0, 0, 0, timestep_start_idx:timestep_start_idx + num_timesteps]
    elif attention_scores.dim() == 3:  # [heads, query_len, key_len]
        cls_attention = attention_scores[0, 0, timestep_start_idx:timestep_start_idx + num_timesteps]
    elif attention_scores.dim() == 2:  # [query_len, key_len]
        cls_attention = attention_scores[0, timestep_start_idx:timestep_start_idx + num_timesteps]
    else:
        print(f" {attention_scores.dim()}")
        return None

    cls_attention = cls_attention.detach().cpu().numpy()

    time_steps = np.arange(num_timesteps)

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, cls_attention, marker='o', linestyle='-', color='b')
    plt.xlabel('observation time t')
    plt.ylabel('CLS attention scores')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, num_timesteps, step=1))

    img_path = os.path.join(save_dir, f"cls_attention_block_{block_idx}.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    return img_path

def main():
    global ANALYSIS_MODE

    device = get_device(DEVICE_IDS, allow_cpu=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    attention_dir = os.path.join(SAVE_DIR, f"{ANALYSIS_MODE}_attention_scores")
    os.makedirs(attention_dir, exist_ok=True)

    config = read_yaml(CFG_PATH)
    config["local_device_ids"] = DEVICE_IDS

    dataloaders = get_dataloaders(config)

    normalize_obj = None
    for t in dataloaders["train"].dataset.transform.transforms:
        if isinstance(t, Normalize):
            t.compute_stats = True
            normalize_obj = t
            break

    if normalize_obj is None:
        raise RuntimeError("Normalize 实例未找到，请检查 transform 列表。")

    with torch.no_grad():
        for _ in tqdm(dataloaders["train"], desc="计算均值/标准差"):
            pass
    normalize_obj.compute_mean_std()
    normalize_obj.compute_stats = False

    mean = normalize_obj.mean.numpy().squeeze()
    std = normalize_obj.std.numpy().squeeze()

    net = get_model(config, device)
    load_from_checkpoint(net, WEIGHTS_PATH, device)
    net.to(device).eval()

    patch_size = getattr(net, 'patch_size', 16)

    if PICKLE_FILE and os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, 'rb') as f:
            data = pickle.load(f)
        img_data, labels, doys = data['img'], data['labels'], data['doy']


        normalized_img = custom_normalize(img_data, mean, std)

        model_input = prepare_model_input(normalized_img, doys)

        T, C, H, W = model_input.shape

        inputs = torch.tensor(model_input, dtype=torch.float32)
        inputs = inputs.unsqueeze(0)  
        inputs = inputs.to(device)

        net.set_analysis_mode(ANALYSIS_MODE)

        with torch.no_grad():
            logits, attention_scores_list = modified_forward_with_attention(inputs, net, mode=ANALYSIS_MODE)

        num_spatial_blocks = (H // patch_size) * (W // patch_size)
        print(f"空间块数量: {num_spatial_blocks}")

        if ANALYSIS_MODE == 'spatial' and hasattr(net, 'gate_weights'):
            spatial_gate_weights = net.gate_weights

            save_gate_weights(
                spatial_gate_weights,
                attention_dir,
                "spatial",
                num_positions=num_spatial_blocks
            )

            plot_gate_weights(
                spatial_gate_weights,
                attention_dir,
                "spatial",
                num_positions=num_spatial_blocks,
                figsize=(12, 6)
            )
        elif ANALYSIS_MODE == 'temporal' and hasattr(net, 'gate_weights'):
            temporal_gate_weights = net.gate_weights

            save_gate_weights(
                temporal_gate_weights,
                attention_dir,
                "temporal",
                num_positions=T 
            )
        else:

        if ANALYSIS_MODE == 'spatial' and attention_scores_list:

            block_importance_per_layer = compute_spatial_block_importance_per_layer(
                attention_scores_list,
                num_blocks=num_spatial_blocks
            )

            if block_importance_per_layer:
                importance_df = pd.DataFrame(
                    block_importance_per_layer,
                    columns=[f"Block_{i}" for i in range(1, num_spatial_blocks + 1)]
                )

                importance_df.insert(0, 'Layer', range(1, len(block_importance_per_layer) + 1))

                importance_path = os.path.join(SAVE_DIR, "spatial_block_importance_per_layer.csv")
                importance_df.to_csv(importance_path, index=False)

        for block_idx, attention_scores in enumerate(attention_scores_list):

            if ANALYSIS_MODE == 'temporal':
                plot_temporal_attention(attention_scores, attention_dir, block_idx)
                plot_attention_over_time(attention_scores, attention_dir, block_idx)
            else:
                plot_spatial_attention(attention_scores, attention_dir, block_idx)
    else:
        print(f"⚠️ 未找到样本文件 {PICKLE_FILE}，请检查路径")

if __name__ == "__main__":

    main()
