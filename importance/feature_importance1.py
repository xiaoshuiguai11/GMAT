import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import get_dataloaders
from data.PASTIS24.data_transforms import Normalize
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint

BAND_NAMES = [
    'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A',
    'EVI', 'GCVI', 'GNDVI', 'NDVI', 'NDWI', 'NREDI1', 'NREDI2', 'NREDI3',
    'OSAVI', 'RVI'
]

def load_single_sample(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
shape}")

    return data['img'], data['labels'], data['doy']

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
    doy_channel = np.broadcast_to(doy_channel,
                                  (doy_normalized.shape[0], 1,
                                   normalized_img.shape[2], normalized_img.shape[3]))
    model_input = np.concatenate([normalized_img, doy_channel], axis=1)

    return model_input.astype(np.float32)

def plot_original_inputs(inputs, band_names, save_path):

    plt.figure(figsize=(14, 8))

    for i in range(min(len(band_names), inputs.shape[1])):
        plt.plot(inputs[:, i], label=band_names[i], linewidth=2)

    plt.title('原始输入值', fontsize=16)
    plt.xlabel('时间步', fontsize=14)
    plt.ylabel('归一化值', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"原始输入值图已保存至: {save_path}")
    plt.close()

def plot_gradients(grads, band_names, save_path):
    plt.figure(figsize=(8, 2))

    for i in range(min(len(band_names), grads.shape[1])):
        plt.plot(grads[:, i], label=band_names[i], linewidth=1)

    num_timesteps = grads.shape[0]
    plt.xticks(np.arange(0, num_timesteps, 2))

    plt.title('输入梯度重要性', fontsize=16)
    plt.xlabel('时间步', fontsize=6)
    plt.ylabel('梯度值', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"梯度重要性图已保存至: {save_path}")
    plt.close()

def process_time_features(xt, device):
    xt = (xt * 365).to(torch.int64)
    xt = torch.clamp(xt, 0, 365)

    max_val = xt.max().item()
    if max_val >= 366:
        xt = torch.clamp(xt, 0, 365)

    xt = F.one_hot(xt, num_classes=366).to(torch.float32)
    return xt

def create_full_target_map(logits, labels, device):

    target = torch.zeros_like(logits).to(device)
    _, num_classes, H, W = logits.shape
    if labels.shape[0] != H or labels.shape[1] != W:
        labels = labels[:H, :W]  
    for y in range(H):
        for x in range(W):
            class_idx = labels[y, x]
            if class_idx < num_classes:
                target[0, class_idx, y, x] = 1.0
    return target

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

    x = x.view(B, T, 20, num_patches_h, net.patch_size, num_patches_w, net.patch_size)
    x = x.permute(0, 3, 5, 1, 4, 6, 2)  # [B, num_patches_h, num_patches_w, T, patch_size, patch_size, C]
    x = x.reshape(B * num_patches_h * num_patches_w, T, net.patch_size * net.patch_size * 20)

    x = net.to_patch_embedding[1](x) 

    x = x.reshape(B, num_patches, T, net.dim)
    x += temporal_pos_embedding.unsqueeze(1)
    x = x.reshape(B * num_patches, T, net.dim)

    cls_temporal_tokens = net.temporal_token.repeat(B * num_patches, 1, 1)
    x = torch.cat((cls_temporal_tokens, x), dim=1)

    x = net.temporal_transformer(x)
    x = x[:, :net.num_classes]

    x = x.reshape(B, num_patches, net.num_classes, net.dim).permute(0, 2, 1, 3).reshape(B * net.num_classes,
                                                                                        num_patches, net.dim)

    space_pos_embedding = net.space_pos_embedding[:, :num_patches] if net.space_pos_embedding.shape[
                                                                          1] > num_patches else net.space_pos_embedding
    x += space_pos_embedding

    if hasattr(net, 'dropout'):
        x = net.dropout(x)

    x = net.space_transformer(x)

    x = net.mlp_head(x.reshape(-1, net.dim))

    x = x.reshape(B, net.num_classes, num_patches, net.patch_size ** 2)
    x = x.permute(0, 2, 3, 1)  # [B, num_patches, patch_size**2, num_classes]

    x = x.reshape(B, num_patches_h, num_patches_w, net.patch_size, net.patch_size, net.num_classes)

    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, num_patches_h, patch_size, num_patches_w, patch_size, num_classes]
    x = x.reshape(B, num_patches_h * net.patch_size, num_patches_w * net.patch_size, net.num_classes)

    x = x.permute(0, 3, 1, 2)  # [B, num_classes, H, W]
    return x


def main(cfg_path, weights_path, device_ids, save_dir, pickle_file_path):

    device = get_device(device_ids, allow_cpu=False)

    os.makedirs(save_dir, exist_ok=True)

    config = read_yaml(cfg_path)
    config["local_device_ids"] = device_ids

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
    load_from_checkpoint(net, weights_path, device)
    net.to(device).eval()
    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    print(f"🚀 加载样本: {pickle_file_path}")
    img_data, labels, doys = load_single_sample(pickle_file_path)

    normalized_img = custom_normalize(img_data, mean, std)

    model_input = prepare_model_input(normalized_img, doys)
    inputs = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, C, H, W]
    inputs.requires_grad = True

    logits = modified_forward(inputs, net.module if hasattr(net, 'module') else net)


    target = create_full_target_map(logits, labels, device)

    probs = F.softmax(logits, dim=1)

    loss = - (target * torch.log(probs + 1e-10)).sum()

    scaled_loss = loss * 1000

    net.zero_grad()
    scaled_loss.backward()

    grads = inputs.grad.detach().cpu().numpy()[0] / 1000  # [T, C, H, W]

    grads_spatial_avg = grads.mean(axis=(2, 3))  # [T, C]

    grads_spatial_avg = grads_spatial_avg[:, :20]

    feature_importance_path = os.path.join(save_dir, "feature_importance.csv")
    df = pd.DataFrame(grads_spatial_avg, columns=BAND_NAMES)
    df['TimeStep'] = range(1, len(df) + 1)
    df.set_index('TimeStep', inplace=True)
    df.to_csv(feature_importance_path)

    center_inputs = normalized_img[:, :, labels.shape[0] // 2, labels.shape[1] // 2]  # 使用标签图的中心

    input_plot_path = os.path.join(save_dir, "original_inputs_plot.png")
    plot_original_inputs(center_inputs, BAND_NAMES, input_plot_path)

    gradient_plot_path = os.path.join(save_dir, "gradients_plot.png")
    plot_gradients(grads_spatial_avg, BAND_NAMES, gradient_plot_path)

    np.save(os.path.join(save_dir, "original_inputs.npy"), img_data)
    np.save(os.path.join(save_dir, "normalized_inputs.npy"), normalized_img)
    np.save(os.path.join(save_dir, "gradients.npy"), grads)

if __name__ == "__main__":
    cfg_path = r"C:\Users\Think\Desktop\GMAT\configs\PASTIS24\TSViT_fold5.yaml"
    weights_path = r"C:\Users\Think\Desktop\模型\logs\门控自适应\best.pth"
    pickle_file_path = r"C:\Users\Think\Desktop\bq\bq_new_new\kuochong_30\64\total2\20369_0_0.pickle"
    save_dir = r"C:\Users\Think\Desktop\feature_importance_results"
    device_ids = [0]

    torch.cuda.init()
    if not torch.cuda.is_initialized():
        print("⚠️ CUDA未正确初始化，尝试使用CPU")
        device_ids = [] 

    main(
        cfg_path=cfg_path,
        weights_path=weights_path,
        device_ids=device_ids,
        save_dir=save_dir,
        pickle_file_path=pickle_file_path

    )
