
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import get_dataloaders
from data.PASTIS24.data_transforms import Normalize
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint


CFG_PATH = r"C:\Users\Think\Desktop\GMAT\configs\PASTIS24\TSViT_fold5.yaml"
WEIGHTS_PATH = r"C:\Users\Think\Desktop\模型\logs\门控自适应\best.pth"
SAVE_DIR = r"C:\Users\Think\Desktop\gate"
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

def modified_forward(inputs, net):

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

    x = x.permute(0, 3, 4, 1, 2, 5, 6)  # [B, num_patches_h, num_patches_w, T, 20, patch_size, patch_size]

    x = x.reshape(B * num_patches_h * num_patches_w, T, 20 * net.patch_size * net.patch_size)

    x = net.to_patch_embedding[1](x)  # [B*num_patches, T, dim]

    x = x.reshape(B, num_patches, T, net.dim)  # [B, num_patches, T, dim]
    x += temporal_pos_embedding.unsqueeze(1)  # [B, num_patches, T, dim]
    x = x.reshape(B * num_patches, T, net.dim)  # [B*num_patches, T, dim]

    cls_temporal_tokens = net.temporal_token.repeat(B * num_patches, 1, 1)
    x = torch.cat((cls_temporal_tokens, x), dim=1)  # [B*num_patches, T+num_classes, dim]

    net.temporal_transformer.collect_gate_weights = True

    x = net.temporal_transformer(x)
    x = x[:, :net.num_classes]  # [B*num_patches, num_classes, dim]

    x = x.reshape(B, num_patches, net.num_classes, net.dim)  # [B, num_patches, num_classes, dim]
    x = x.permute(0, 2, 1, 3)  # [B, num_classes, num_patches, dim]
    x = x.reshape(B * net.num_classes, num_patches, net.dim)  # [B*num_classes, num_patches, dim]

    space_pos_embedding = net.space_pos_embedding[:, :num_patches] if net.space_pos_embedding.shape[
                                                                          1] > num_patches else net.space_pos_embedding
    x += space_pos_embedding

    if hasattr(net, 'dropout'):
        x = net.dropout(x)

    if hasattr(net, 'collect_gate_weights'):
        net.collect_gate_weights = True
    if hasattr(net.space_transformer, 'collect_gate_weights'):
        net.space_transformer.collect_gate_weights = True

    x = net.space_transformer(x)  # [B*num_classes, num_patches, dim]

    x = net.mlp_head(x.reshape(-1, net.dim))  # [B*num_classes*num_patches, patch_size**2]

    x = x.reshape(B, net.num_classes, num_patches, net.patch_size ** 2)  # [B, num_classes, num_patches, patch_size**2]
    x = x.permute(0, 2, 3, 1)  # [B, num_patches, patch_size**2, num_classes]

    x = x.reshape(B, num_patches_h, num_patches_w, net.patch_size, net.patch_size, net.num_classes)

    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, num_patches_h, patch_size, num_patches_w, patch_size, num_classes]
    x = x.reshape(B, num_patches_h * net.patch_size, num_patches_w * net.patch_size, net.num_classes)

    x = x.permute(0, 3, 1, 2)  # [B, num_classes, H, W]
    return x


def plot_gate_weights(gate_weights, save_dir, block_idx, mode='spatial'):
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)
    mode_dir = os.path.join(save_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)

    attn_values = gate_weights['attn_weights']  # [S, D]
    mamba_values = gate_weights['mamba_weights']  # [S, D]

    attn_df = pd.DataFrame(attn_values)
    mamba_df = pd.DataFrame(mamba_values)

    attn_csv = os.path.join(mode_dir, f'attn_gate_weights_block_{block_idx}.csv')
    mamba_csv = os.path.join(mode_dir, f'mamba_gate_weights_block_{block_idx}.csv')
    attn_df.to_csv(attn_csv, index=False)
    mamba_df.to_csv(mamba_csv, index=False)

    for name, values in zip(["Attention", "Mamba"], [attn_values, mamba_values]):
        plt.figure(figsize=(12, 6))
        sns.heatmap(values, cmap="viridis", cbar=True)
        plt.title(f'{name} Gate Weights Heatmap - Block {block_idx}', fontsize=14)
        plt.xlabel('Channel Dimension (D)', fontsize=12)
        plt.ylabel('Patch Index (S)', fontsize=12)
        save_path = os.path.join(mode_dir, f'{name.lower()}_gate_weights_block_{block_idx}_heatmap.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

def process_weights(weights_dict, mode='temporal'):
    attn_weights = weights_dict['attn']  # [B, L, D]
    mamba_weights = weights_dict['mamba']

    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.cpu().numpy()
    if isinstance(mamba_weights, torch.Tensor):
        mamba_weights = mamba_weights.cpu().numpy()

    if mode == 'temporal' and attn_weights.shape[1] > 4:
        attn_weights = attn_weights[:, 4:, :]
        mamba_weights = mamba_weights[:, 4:, :]

    if mode == 'temporal':
        attn_avg = np.mean(attn_weights, axis=(0, 2))
        mamba_avg = np.mean(mamba_weights, axis=(0, 2))
    else:

        attn_avg = np.mean(attn_weights, axis=0)  # → shape: [S, D]
        mamba_avg = np.mean(mamba_weights, axis=0)

    return {
        'attn_weights': attn_avg,
        'mamba_weights': mamba_avg
    }

def main():

    device = get_device(DEVICE_IDS, allow_cpu=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    gate_weights_dir = os.path.join(SAVE_DIR, "gate_weights")
    os.makedirs(gate_weights_dir, exist_ok=True)

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
    print(f"✅ 归一化统计完成: mean={mean}, std={std}")

    net = get_model(config, device)
    load_from_checkpoint(net, WEIGHTS_PATH, device)
    net.to(device).eval()

    if hasattr(net, 'set_analysis_mode'):
        net.set_analysis_mode(ANALYSIS_MODE)

    patch_size = getattr(net, 'patch_size', 16)

    if PICKLE_FILE and os.path.exists(PICKLE_FILE):

        with open(PICKLE_FILE, 'rb') as f:
            data = pickle.load(f)
        img_data, labels, doys = data['img'], data['labels'], data['doy']

        normalized_img = custom_normalize(img_data, mean, std)

        model_input = prepare_model_input(normalized_img, doys)

        T, C, H, W = model_input.shape

        inputs = torch.tensor(model_input, dtype=torch.float32)  # [T, C, H, W]
        inputs = inputs.unsqueeze(0)  # 添加批次维度 [1, T, C, H, W]
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = modified_forward(inputs, net)

        if ANALYSIS_MODE == 'temporal':
            if hasattr(net.temporal_transformer, 'gate_weights') and net.temporal_transformer.gate_weights:
                gate_weights = net.temporal_transformer.gate_weights

                for block_idx, weights_dict in enumerate(gate_weights):
                    processed_weights = process_weights(weights_dict, mode='temporal')

                    plot_gate_weights(processed_weights, gate_weights_dir, block_idx, mode='temporal')
            else:
                print("⚠️ 警告: 时间转换器没有'gate_weights'属性或该属性为空")
        else:
            if hasattr(net.space_transformer, 'gate_weights') and net.space_transformer.gate_weights:
                gate_weights = net.space_transformer.gate_weights

                for block_idx, weights_dict in enumerate(gate_weights):
                    processed_weights = process_weights(weights_dict, mode='spatial')

                    plot_gate_weights(processed_weights, gate_weights_dir, block_idx, mode='spatial')
            else:
                print("⚠️ 警告: 空间转换器没有'gate_weights'属性或该属性为空")
    else:
        if PICKLE_FILE:
            print(f"⚠️ 未找到样本文件 {PICKLE_FILE}，改用验证集")

        val_loader = dataloaders["eval"]
        sample_count = 0

        progress = tqdm(total=NUM_SAMPLES, desc="处理样本")

        for inputs, labels in val_loader:
            if sample_count >= NUM_SAMPLES:
                break

            inputs = inputs.to(device)

            with torch.no_grad():
                logits = net(inputs)

            if ANALYSIS_MODE == 'temporal':
                # 时间模式
                if hasattr(net.temporal_transformer, 'gate_weights') and net.temporal_transformer.gate_weights:
                    gate_weights = net.temporal_transformer.gate_weights

                    for sample_idx in range(inputs.size(0)):
                        if sample_count >= NUM_SAMPLES:
                            break
                        for block_idx, weights_dict in enumerate(gate_weights):
                            processed_weights = process_weights(weights_dict, mode='temporal')
                            plot_gate_weights(processed_weights, gate_weights_dir, block_idx, mode='temporal')

                        sample_count += 1
                        progress.update(1)
                else:
                    break
            else:

                if hasattr(net.space_transformer, 'gate_weights') and net.space_transformer.gate_weights:
                    gate_weights = net.space_transformer.gate_weights

                    for sample_idx in range(inputs.size(0)):
                        if sample_count >= NUM_SAMPLES:
                            break

                        for block_idx, weights_dict in enumerate(gate_weights):
                            processed_weights = process_weights(weights_dict, mode='spatial')

                            plot_gate_weights(processed_weights, gate_weights_dir, block_idx, mode='spatial')

                        sample_count += 1
                        progress.update(1)
                else:
                    break

        progress.close()


if __name__ == "__main__":

    main()
