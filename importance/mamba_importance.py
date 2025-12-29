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
import warnings
from scipy.special import softmax
import math

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import get_dataloaders
from data.PASTIS24.data_transforms import Normalize
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint


CFG_PATH = r"C:\Users\Think\Desktop\GMAT\configs\PASTIS24\TSViT_fold5.yaml"
WEIGHTS_PATH = r"C:\Users\Think\Desktop\模型\logs\门控自适应\best.pth"
SAVE_DIR = r"C:\Users\Think\Desktop\gate_analysis"
PICKLE_FILE = r"C:\Users\Think\Desktop\bq\bq_new_new\kuochong_30\64\total2\20369_1_0.pickle"
NUM_SAMPLES = 5
DEVICE_IDS = [0]
ANALYSIS_MODE = 'spatial'  

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
        xt = torch.clamp(xt, 0, 365)
    xt = F.one_hot(xt, num_classes=366).to(torch.float32)
    return xt

def plot_gate_weights_line(gate_data, save_path):

    plt.figure(figsize=(12, 8))

    block_indices = [d[0] for d in gate_data]
    attn_means = [d[1] for d in gate_data]
    mamba_means = [d[2] for d in gate_data]

    plt.plot(block_indices, attn_means, marker='o', linestyle='-', color='blue', label='Attention Branch')
    plt.plot(block_indices, mamba_means, marker='s', linestyle='-', color='green', label='Mamba Branch')

    plt.title('平均门控权重随块变化趋势', fontsize=16)
    plt.xlabel('块索引', fontsize=12)
    plt.ylabel('平均权重值', fontsize=12)
    plt.xticks(block_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gate_weights_box(gate_data, save_path):

    plt.figure(figsize=(15, 8))

    data_to_plot = []
    labels = []
    for block_idx, weights in gate_data:
        data_to_plot.append(weights)
        labels.append(f'块 {block_idx}')

    plt.boxplot(data_to_plot, labels=labels, showfliers=False)

    plt.title('门控权重分布随块变化', fontsize=16)
    plt.xlabel('块索引', fontsize=12)
    plt.ylabel('权重值', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def plot_single_feature_importance(block_idx, feature_imp, save_dir):

    if len(feature_imp.shape) > 1:
        feature_imp = np.mean(feature_imp, axis=0) 

    normalized_imp = softmax(feature_imp)

    plt.figure(figsize=(15, 4))
    sns.heatmap(
        normalized_imp.reshape(1, -1),
        cmap='viridis',
        cbar=True,
        annot=False,
        yticklabels=False
    )
    plt.title(f'块 {block_idx} 特征关注度')
    plt.xlabel('特征维度')

    save_path = os.path.join(save_dir, f"feature_importance_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_timestep_importance(block_idx, timestep_imp, save_dir):

    if len(timestep_imp.shape) > 1:
        timestep_imp = np.mean(timestep_imp, axis=0)  # 沿空间位置平均

    normalized_imp = softmax(timestep_imp)
    print(f"normalized_imp shape after softmax: {normalized_imp.shape}")

    plt.figure(figsize=(15, 4))
    sns.heatmap(
        normalized_imp.reshape(1, -1), 
        cmap='magma',
        cbar=True,
        annot=False,
        yticklabels=False
    )
    plt.title(f'块 {block_idx} 时间步关注度')
    plt.xlabel('时间步')

    save_path = os.path.join(save_dir, f"timestep_importance_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_single_feature_line(block_idx, feature_imp, save_dir):

    if len(feature_imp.shape) > 1:
        feature_imp = np.mean(feature_imp, axis=0)  # 沿空间位置平均

    plt.figure(figsize=(15, 6))

    normalized_imp = softmax(feature_imp)

    plt.plot(normalized_imp, marker='o', linestyle='-', color='blue')

    plt.title(f'块 {block_idx} 特征关注度分布', fontsize=16)
    plt.xlabel('特征维度索引', fontsize=12)
    plt.ylabel('关注度值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(save_dir, f"feature_importance_line_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
def plot_single_timestep_line(block_idx, timestep_imp, save_dir):

    if len(timestep_imp.shape) > 1:
        timestep_imp = np.mean(timestep_imp, axis=0)  # 沿空间位置平均


    normalized_imp = softmax(timestep_imp)
    print(f"normalized_imp shape after softmax: {normalized_imp.shape}")

    csv_path = os.path.join(save_dir, f"timestep_importance_block_{block_idx}.csv")
    df = pd.DataFrame({
        'timestep_index': range(len(normalized_imp)),
        'normalized_importance': normalized_imp
    })
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(15, 6))
    plt.plot(normalized_imp, marker='s', linestyle='-', color='green')
    plt.title(f'块 {block_idx} 时间步关注度分布', fontsize=16)
    plt.xlabel('时间步索引', fontsize=12)
    plt.ylabel('关注度值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(save_dir, f"timestep_importance_line_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_space_importance(block_idx, space_imp, save_dir, grid_size, patch_size, image_size):
    
    expected_size = grid_size * grid_size
    if space_imp.size != expected_size:
        print(f"⚠️ 警告: 空间位置数据大小 {space_imp.size} 与预期网格大小 {expected_size} 不匹配")
        return None

    normalized_imp =space_imp

    grid_imp = normalized_imp.reshape(grid_size, grid_size)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        grid_imp,
        cmap='viridis',
        annot=False,
        square=True,
        cbar=True,
        cbar_kws={'label': '关注度'}
    )
    plt.title(f'空间位置关注度 (块 {block_idx})', fontsize=16)
    plt.xlabel('X 位置', fontsize=12)
    plt.ylabel('Y 位置', fontsize=12)

    save_path = os.path.join(save_dir, f"space_importance_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    csv_path = os.path.join(save_dir, f"space_importance_block_{block_idx}.csv")
    np.savetxt(csv_path, grid_imp, delimiter=",")
    return grid_imp


def plot_pixel_importance(block_idx, grid_imp, save_dir, grid_size, patch_size, image_size):

    pixel_imp = np.zeros((image_size, image_size))

    for i in range(grid_size):
        for j in range(grid_size):
            start_h = i * patch_size
            end_h = min((i + 1) * patch_size, image_size)
            start_w = j * patch_size
            end_w = min((j + 1) * patch_size, image_size)

            pixel_imp[start_h:end_h, start_w:end_w] = grid_imp[i, j]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pixel_imp,
        cmap='viridis',
        annot=False,
        square=False,
        cbar=True,
        cbar_kws={'label': '像素关注度'}
    )
    plt.title(f'像素级空间位置关注度 (块 {block_idx})', fontsize=18)
    plt.xlabel('X 像素位置', fontsize=14)
    plt.ylabel('Y 像素位置', fontsize=14)

    save_path = os.path.join(save_dir, f"pixel_importance_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    csv_path = os.path.join(save_dir, f"pixel_importance_block_{block_idx}.csv")
    np.savetxt(csv_path, pixel_imp, delimiter=",")

    max_imp = np.max(pixel_imp)
    threshold = max_imp * 0.7  # 70%阈值
    high_imp_coords = np.argwhere(pixel_imp > threshold)

    if len(high_imp_coords) > 0:

        min_h, min_w = np.min(high_imp_coords, axis=0)
        max_h, max_w = np.max(high_imp_coords, axis=0)
        
    return pixel_imp


def plot_space_position_importance_line(space_data, save_path):

    plt.figure(figsize=(12, 8))

    block_indices = [d[0] for d in space_data]
    avg_importance = [d[1] for d in space_data]

    plt.plot(block_indices, avg_importance, marker='o', linestyle='-', color='purple', label='平均关注度')

    plt.title('空间位置平均关注度随块变化趋势', fontsize=16)
    plt.xlabel('块索引', fontsize=12)
    plt.ylabel('平均关注度值', fontsize=12)
    plt.xticks(block_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def modified_forward_with_gate_weights(inputs, net, analysis_mode):

    B, T, C, H, W = inputs.shape
    inputs = inputs.float()

    net.set_analysis_mode(analysis_mode)
    print(f"✅ 设置分析模式: {analysis_mode}")

    xt = inputs[:, :, -1, 0, 0]
    xt = process_time_features(xt, inputs.device)
    xt = xt.reshape(-1, 366)

    temporal_pos_embedding = net.to_temporal_embedding_input(xt).reshape(B, T, net.dim)

    x = inputs[:, :, :-1]  # 移除时间特征通道

    assert H % net.patch_size == 0, f"高度 {H} 不能被 patch_size {net.patch_size} 整除"
    assert W % net.patch_size == 0, f"宽度 {W} 不能被 patch_size {net.patch_size} 整除"

    num_patches_h = H // net.patch_size
    num_patches_w = W // net.patch_size
    num_patches = num_patches_h * num_patches_w

    grid_size = num_patches_h

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

    gate_weights_data = []

    feature_importance_data = []
    timestep_importance_data = []

    position_importance_data = []

    for block in net.temporal_transformer.layers:

        block.feature_importance = None
        block.timestep_importance = None
        block.attention_scores = None

    x = net.temporal_transformer(x)

    for block_idx, block in enumerate(net.temporal_transformer.layers):

        if hasattr(block, 'attn_weights') and block.attn_weights:
            attn_weights_np = block.attn_weights[-1].flatten()
            gate_weights_data.append((block_idx, attn_weights_np))

        if hasattr(block, 'feature_importance') and block.feature_importance is not None:
            feat_imp = block.feature_importance.detach().cpu().numpy()
            feature_importance_data.append((block_idx, feat_imp))

        if hasattr(block, 'timestep_importance') and block.timestep_importance is not None:

            timestep_imp = block.timestep_importance.detach().cpu().numpy()

            if timestep_imp.shape[1] > 4:  
                timestep_imp = timestep_imp[:, 4:] 
            timestep_importance_data.append((block_idx, timestep_imp))

    if analysis_mode == 'temporal':
        temporal_pos_imp = net.get_temporal_position_importance()
        if temporal_pos_imp:
            for block_idx, imp in enumerate(temporal_pos_imp):
                if isinstance(imp, torch.Tensor):
                    imp = imp.detach().cpu().numpy()
                position_importance_data.append((block_idx, imp))
        else:
            print("⚠️ 未检测到时间位置重要性数据")

    x = x[:, :net.num_classes]
    x = x.reshape(B, num_patches, net.num_classes, net.dim)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B * net.num_classes, num_patches, net.dim)

    space_pos_embedding = net.space_pos_embedding[:, :num_patches] if net.space_pos_embedding.shape[
                                                                          1] > num_patches else net.space_pos_embedding
    x += space_pos_embedding

    if hasattr(net, 'dropout'):
        x = net.dropout(x)

    x = net.space_transformer(x)

    if analysis_mode == 'spatial' and hasattr(net.space_transformer, 'get_space_position_importance'):
        space_pos_imp = net.space_transformer.get_space_position_importance()
        if space_pos_imp:
            for block_idx, imp in enumerate(space_pos_imp):

                if isinstance(imp, torch.Tensor):
                    imp = imp.detach().cpu().numpy()
                position_importance_data.append((block_idx, imp))
        else:
            print("⚠️ 空间位置重要性数据为空")

    x = net.mlp_head(x.reshape(-1, net.dim))

    x = x.reshape(B, net.num_classes, num_patches, net.patch_size ** 2)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(B, num_patches_h, num_patches_w, net.patch_size, net.patch_size, net.num_classes)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, num_patches_h * net.patch_size, num_patches_w * net.patch_size, net.num_classes)
    x = x.permute(0, 3, 1, 2)

    return x, gate_weights_data, feature_importance_data, timestep_importance_data, position_importance_data, grid_size


def save_spatial_patch_importance(position_importance_data, save_dir, num_patches):
    
    summary_df = pd.DataFrame()

    for block_idx, imp_vec in position_importance_data:
    
        if len(imp_vec) < num_patches:
            padded_vec = np.zeros(num_patches)
            padded_vec[:len(imp_vec)] = imp_vec
            imp_vec = padded_vec
        elif len(imp_vec) > num_patches:
            imp_vec = imp_vec[:num_patches]

        df = pd.DataFrame({
            'patch_index': range(num_patches),
            'importance': imp_vec
        })

        csv_path = os.path.join(save_dir, f"spatial_patch_importance_block_{block_idx}.csv")
        df.to_csv(csv_path, index=False)

        summary_df[f'block_{block_idx}'] = imp_vec

    if not summary_df.empty:
        summary_df.insert(0, 'patch_index', range(num_patches))
        summary_csv_path = os.path.join(save_dir, "spatial_patch_importance_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
    else:
        print("⚠️ 未生成空间patch重要性汇总文件，无有效数据")


def main():
    global ANALYSIS_MODE

    device = get_device(DEVICE_IDS, allow_cpu=True)

    os.makedirs(SAVE_DIR, exist_ok=True)

    if ANALYSIS_MODE == 'temporal':
        main_dir = os.path.join(SAVE_DIR, "temporal_analysis")
    elif ANALYSIS_MODE == 'spatial':
        main_dir = os.path.join(SAVE_DIR, "spatial_analysis")
    else:
        raise ValueError(f"无效的分析模式: {ANALYSIS_MODE}")

    os.makedirs(main_dir, exist_ok=True)

    gate_dir = os.path.join(main_dir, "gate_weights_analysis")
    feature_dir = os.path.join(main_dir, "feature_importance")
    timestep_dir = os.path.join(main_dir, "timestep_importance")
    position_dir = os.path.join(main_dir, "position_importance")
    patch_importance_dir = os.path.join(main_dir, "patch_importance")  

    os.makedirs(gate_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(timestep_dir, exist_ok=True)
    os.makedirs(position_dir, exist_ok=True)
    os.makedirs(patch_importance_dir, exist_ok=True)  

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
        print(f"🚀 加载样本: {PICKLE_FILE}")
        with open(PICKLE_FILE, 'rb') as f:
            data = pickle.load(f)
        img_data, labels, doys = data['img'], data['labels'], data['doy']

        normalized_img = custom_normalize(img_data, mean, std)

        model_input = prepare_model_input(normalized_img, doys)

        T, C, H, W = model_input.shape

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w

        inputs = torch.tensor(model_input, dtype=torch.float32)
        inputs = inputs.unsqueeze(0)  # 添加批次维度 [1, T, C, H, W]
        inputs = inputs.to(device)

        with torch.no_grad():
            logits, gate_weights_data, feature_importance_data, timestep_importance_data, position_importance_data, grid_size = \
                modified_forward_with_gate_weights(inputs, net, ANALYSIS_MODE)

        if gate_weights_data:

            line_plot_data = []
            for block_idx, weights in gate_weights_data:
                attn_mean = np.mean(weights)  # 注意力分支平均权重
                mamba_mean = 1 - attn_mean  # Mamba分支平均权重
                line_plot_data.append((block_idx, attn_mean, mamba_mean))

            line_path = os.path.join(gate_dir, "gate_weights_line.png")
            plot_gate_weights_line(line_plot_data, line_path)

            box_path = os.path.join(gate_dir, "gate_weights_box.png")
            plot_gate_weights_box(gate_weights_data, box_path)

            for block_idx, weights in gate_weights_data:
                print(f"Saving gate weights for block {block_idx}, weights shape: {weights.shape}")
                csv_path = os.path.join(gate_dir, f"gate_weights_block_{block_idx}.csv")
                pd.DataFrame(weights, columns=['weight']).to_csv(csv_path, index=False)
        else:
            print("⚠️ 未检测到门控权重数据，请检查模型结构")

        if feature_importance_data:

            for block_idx, feat_imp in feature_importance_data:
                plot_single_feature_importance(block_idx, feat_imp, feature_dir)
                plot_single_feature_line(block_idx, feat_imp, feature_dir)

                csv_path = os.path.join(feature_dir, f"feature_importance_block_{block_idx}.csv")
                pd.DataFrame(feat_imp).to_csv(csv_path, index=False)
        else:
            print("⚠️ 未检测到Mamba特征关注度数据")

        if timestep_importance_data:

            for block_idx, time_imp in timestep_importance_data:
                plot_single_timestep_importance(block_idx, time_imp, timestep_dir)

                plot_single_timestep_line(block_idx, time_imp, timestep_dir)
        else:
            print("⚠️ 未检测到Mamba时间步关注度数据")

        if position_importance_data:

            if ANALYSIS_MODE == 'temporal':
                for block_idx, pos_imp in position_importance_data:
                    plot_single_timestep_line(block_idx, pos_imp, position_dir)

            elif ANALYSIS_MODE == 'spatial':
                save_spatial_patch_importance(
                    position_importance_data=position_importance_data,
                    save_dir=patch_importance_dir,
                    num_patches=num_patches
                )

                line_plot_data = []
                for block_idx, pos_imp in position_importance_data:
                    grid_imp = plot_single_space_importance(
                        block_idx=block_idx,
                        space_imp=pos_imp,
                        save_dir=position_dir,
                        grid_size=grid_size,
                        patch_size=patch_size,
                        image_size=H 
                    )

                    if grid_imp is not None:
                        pixel_imp = plot_pixel_importance(
                            block_idx=block_idx,
                            grid_imp=grid_imp,
                            save_dir=position_dir,
                            grid_size=grid_size,
                            patch_size=patch_size,
                            image_size=H 
                        )

                    avg_imp = np.mean(pos_imp)
                    line_plot_data.append((block_idx, avg_imp))
                if line_plot_data:
                    line_path = os.path.join(position_dir, "space_importance_trend.png")
                    plot_space_position_importance_line(line_plot_data, line_path)
        else:
            print("⚠️ 未检测到位置重要性数据")
    else:
        print(f"⚠️ 未找到样本文件 {PICKLE_FILE}，请检查路径")

if __name__ == "__main__":

    main()
