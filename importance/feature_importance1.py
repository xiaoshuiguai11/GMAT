#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 TSViT 模型在时序遥感数据上的特征重要性（波段 × 时间）。
修复了梯度消失问题，并优化了可视化效果。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys

# --- 路径与包 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import get_dataloaders
from data.PASTIS24.data_transforms import Normalize
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint

# 定义波段名称
BAND_NAMES = [
    'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A',
    'EVI', 'GCVI', 'GNDVI', 'NDVI', 'NDWI', 'NREDI1', 'NREDI2', 'NREDI3',
    'OSAVI', 'RVI'
]


# --- 加载单个样本数据 ---
def load_single_sample(pickle_file_path):
    """从 pickle 文件中加载数据"""
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"加载数据键: {list(data.keys())}")
    print(f"图像形状: {data['img'].shape}")
    print(f"标签形状: {data['labels'].shape}")
    print(f"日期形状: {data['doy'].shape}")

    return data['img'], data['labels'], data['doy']


# --- 自定义归一化处理 ---
def custom_normalize(data, mean, std):
    """手动应用归一化处理，支持任意维度的数据"""
    # 确保均值和标准差的形状与数据通道维度匹配
    mean = mean.squeeze().astype(np.float32)  # 确保为float32
    std = std.squeeze().astype(np.float32)  # 确保为float32

    # 扩展均值和标准差的维度以匹配数据形状
    if data.ndim == 4:  # (T, C, H, W)
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)
    elif data.ndim == 3:  # (C, H, W)
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
    else:
        raise ValueError(f"不支持的输入维度: {data.ndim}")

    # 应用归一化
    normalized = (data - mean) / std
    return normalized.astype(np.float32)  # 确保为float32


# --- 准备模型输入 ---
def prepare_model_input(normalized_img, doys):
    """
    准备符合模型输入的张量
    根据TSViT模型要求，添加时间特征作为额外通道
    """
    # 1. 准备时间特征 (作为额外的通道)
    # 归一化DOY到[0,1]范围
    doy_normalized = doys / 365.0

    # 扩展DOY为通道 [T, 1, H, W]
    doy_channel = doy_normalized[:, np.newaxis, np.newaxis, np.newaxis]
    doy_channel = np.broadcast_to(doy_channel,
                                  (doy_normalized.shape[0], 1,
                                   normalized_img.shape[2], normalized_img.shape[3]))

    # 2. 将时间特征作为额外通道添加
    model_input = np.concatenate([normalized_img, doy_channel], axis=1)

    print(f"模型输入形状: {model_input.shape} (T, C, H, W)")
    return model_input.astype(np.float32)  # 确保为float32


# --- 可视化原始输入值 ---
def plot_original_inputs(inputs, band_names, save_path):
    """绘制原始输入值并保存"""
    plt.figure(figsize=(14, 8))

    # 绘制所有波段的原始输入值
    for i in range(min(len(band_names), inputs.shape[1])):
        plt.plot(inputs[:, i], label=band_names[i], linewidth=2)

    plt.title('原始输入值', fontsize=16)
    plt.xlabel('时间步', fontsize=14)
    plt.ylabel('归一化值', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"原始输入值图已保存至: {save_path}")
    plt.close()


# --- 可视化梯度重要性 ---
def plot_gradients(grads, band_names, save_path):
    """绘制梯度重要性并保存"""
    plt.figure(figsize=(8, 2))

    # 绘制所有波段的梯度重要性
    for i in range(min(len(band_names), grads.shape[1])):
        plt.plot(grads[:, i], label=band_names[i], linewidth=1)

    # 设置横坐标间隔为2
    num_timesteps = grads.shape[0]
    plt.xticks(np.arange(0, num_timesteps, 2))

    plt.title('输入梯度重要性', fontsize=16)
    plt.xlabel('时间步', fontsize=6)
    plt.ylabel('梯度值', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"梯度重要性图已保存至: {save_path}")
    plt.close()


# --- 时间特征处理 ---
def process_time_features(xt, device):
    """处理时间特征，避免索引错误"""
    # 修正：直接乘以365（不要用365.0001）
    xt = (xt * 365).to(torch.int64)
    xt = torch.clamp(xt, 0, 365)

    # 检查最大值是否超过365
    max_val = xt.max().item()
    if max_val >= 366:
        print(f"⚠️ 警告: 最大时间特征值 {max_val} 超过365，将被裁剪")
        xt = torch.clamp(xt, 0, 365)

    # 执行one-hot编码
    xt = F.one_hot(xt, num_classes=366).to(torch.float32)
    return xt


# --- 创建完整目标图 ---
def create_full_target_map(logits, labels, device):
    """
    创建完整的目标热力图（使用真实标签）
    logits: [1, num_classes, H, W]
    labels: [H, W] (numpy array)
    """
    target = torch.zeros_like(logits).to(device)
    _, num_classes, H, W = logits.shape
    # 确保标签形状匹配
    if labels.shape[0] != H or labels.shape[1] != W:
        labels = labels[:H, :W]  # 裁剪到相同空间尺寸
    for y in range(H):
        for x in range(W):
            class_idx = labels[y, x]
            if class_idx < num_classes:  # 确保不越界
                target[0, class_idx, y, x] = 1.0
    return target


# --- 修改后的前向传播 ---
def modified_forward(inputs, net):
    """修改后的模型前向传播函数，修复数据类型和形状问题"""
    # 提取输入形状
    B, T, C, H, W = inputs.shape

    # 确保输入是float32类型
    inputs = inputs.float()

    # 提取时间特征 (最后一个通道)
    xt = inputs[:, :, -1, 0, 0]
    xt = process_time_features(xt, inputs.device)
    xt = xt.reshape(-1, 366)

    # 应用时间位置嵌入
    temporal_pos_embedding = net.to_temporal_embedding_input(xt).reshape(B, T, net.dim)

    # 准备patch嵌入
    x = inputs[:, :, :-1]  # 移除时间特征通道，保留20个波段 [B, T, 20, H, W]

    # 确保空间维度能被patch_size整除
    assert H % net.patch_size == 0, f"高度 {H} 不能被 patch_size {net.patch_size} 整除"
    assert W % net.patch_size == 0, f"宽度 {W} 不能被 patch_size {net.patch_size} 整除"

    # 计算patch数量
    num_patches_h = H // net.patch_size
    num_patches_w = W // net.patch_size
    num_patches = num_patches_h * num_patches_w

    # 手动实现Rearrange操作
    x = x.view(B, T, 20, num_patches_h, net.patch_size, num_patches_w, net.patch_size)
    x = x.permute(0, 3, 5, 1, 4, 6, 2)  # [B, num_patches_h, num_patches_w, T, patch_size, patch_size, C]
    x = x.reshape(B * num_patches_h * num_patches_w, T, net.patch_size * net.patch_size * 20)

    # 应用线性变换
    x = net.to_patch_embedding[1](x)  # 只应用线性层，跳过Rearrange

    # 添加时间位置嵌入
    x = x.reshape(B, num_patches, T, net.dim)
    x += temporal_pos_embedding.unsqueeze(1)
    x = x.reshape(B * num_patches, T, net.dim)

    # 添加时间token
    cls_temporal_tokens = net.temporal_token.repeat(B * num_patches, 1, 1)
    x = torch.cat((cls_temporal_tokens, x), dim=1)

    # 时间变换器
    x = net.temporal_transformer(x)
    x = x[:, :net.num_classes]

    # 空间变换器
    x = x.reshape(B, num_patches, net.num_classes, net.dim).permute(0, 2, 1, 3).reshape(B * net.num_classes,
                                                                                        num_patches, net.dim)

    # 确保空间位置嵌入大小匹配
    space_pos_embedding = net.space_pos_embedding[:, :num_patches] if net.space_pos_embedding.shape[
                                                                          1] > num_patches else net.space_pos_embedding
    x += space_pos_embedding

    # 应用dropout
    if hasattr(net, 'dropout'):
        x = net.dropout(x)

    # 空间变换器
    x = net.space_transformer(x)

    # MLP头部
    x = net.mlp_head(x.reshape(-1, net.dim))

    # 重塑输出
    x = x.reshape(B, net.num_classes, num_patches, net.patch_size ** 2)
    x = x.permute(0, 2, 3, 1)  # [B, num_patches, patch_size**2, num_classes]

    # 重塑为最终输出形状
    # 首先重塑为 [B, num_patches_h, num_patches_w, patch_size, patch_size, num_classes]
    x = x.reshape(B, num_patches_h, num_patches_w, net.patch_size, net.patch_size, net.num_classes)

    # 然后组合为完整图像
    # 组合高度块
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, num_patches_h, patch_size, num_patches_w, patch_size, num_classes]
    x = x.reshape(B, num_patches_h * net.patch_size, num_patches_w * net.patch_size, net.num_classes)

    # 调整维度顺序
    x = x.permute(0, 3, 1, 2)  # [B, num_classes, H, W]
    return x


# ---------- 主函数 ----------
def main(cfg_path, weights_path, device_ids, save_dir, pickle_file_path):
    # 0. 设备
    device = get_device(device_ids, allow_cpu=False)

    # 1. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 2. 读取配置
    config = read_yaml(cfg_path)
    config["local_device_ids"] = device_ids

    # 3. 创建临时dataloader以获取归一化参数
    print("📊 统计训练集均值 / 标准差 ...")
    dataloaders = get_dataloaders(config)

    # 4. 获取归一化参数
    normalize_obj = None
    for t in dataloaders["train"].dataset.transform.transforms:
        if isinstance(t, Normalize):
            t.compute_stats = True
            normalize_obj = t
            break

    if normalize_obj is None:
        raise RuntimeError("Normalize 实例未找到，请检查 transform 列表。")

    # 计算归一化参数
    with torch.no_grad():
        for _ in tqdm(dataloaders["train"], desc="计算均值/标准差"):
            pass
    normalize_obj.compute_mean_std()
    normalize_obj.compute_stats = False

    # 获取均值和标准差
    mean = normalize_obj.mean.numpy().squeeze()
    std = normalize_obj.std.numpy().squeeze()
    print(f"✅ 归一化统计完成: mean={mean}, std={std}")

    # 5. 构建并加载模型
    net = get_model(config, device)
    load_from_checkpoint(net, weights_path, device)
    net.to(device).eval()
    if len(device_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    # 6. 加载并分析单个样本
    print(f"🚀 加载样本: {pickle_file_path}")
    img_data, labels, doys = load_single_sample(pickle_file_path)

    # 7. 应用自定义归一化
    normalized_img = custom_normalize(img_data, mean, std)
    print(f"归一化后图像形状: {normalized_img.shape}")

    # 8. 准备模型输入
    model_input = prepare_model_input(normalized_img, doys)
    inputs = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, C, H, W]
    inputs.requires_grad = True

    # 9. 前向传播
    print("运行模型前向传播...")

    # 使用修改后的前向传播
    logits = modified_forward(inputs, net.module if hasattr(net, 'module') else net)
    print(f"模型输出形状: {logits.shape}")

    # 10. 计算目标类别得分并反向传播
    # 创建完整的目标图（使用真实标签）
    target = create_full_target_map(logits, labels, device)

    # 关键修改1：使用softmax概率计算损失
    probs = F.softmax(logits, dim=1)

    # 关键修改2：使用交叉熵损失代替点乘损失
    loss = - (target * torch.log(probs + 1e-10)).sum()

    # 关键修改3：梯度放大（解决梯度消失问题）
    scaled_loss = loss * 1000

    # 反向传播
    net.zero_grad()
    scaled_loss.backward()

    # 获取梯度并还原（除以放大倍数）
    grads = inputs.grad.detach().cpu().numpy()[0] / 1000  # [T, C, H, W]
    print(f"梯度形状: {grads.shape}")

    # 打印梯度统计信息
    print(f"梯度范围: {grads.min():.6f} ~ {grads.max():.6f}")
    print(f"梯度均值: {grads.mean():.6f}, 绝对值均值: {np.abs(grads).mean():.6f}")

    # 11. 计算特征重要性 (按时间和波段平均)
    # 空间平均
    grads_spatial_avg = grads.mean(axis=(2, 3))  # [T, C]

    # 只取前20个波段（忽略时间特征通道）
    grads_spatial_avg = grads_spatial_avg[:, :20]

    # 保存特征重要性数据
    feature_importance_path = os.path.join(save_dir, "feature_importance.csv")
    df = pd.DataFrame(grads_spatial_avg, columns=BAND_NAMES)
    df['TimeStep'] = range(1, len(df) + 1)
    df.set_index('TimeStep', inplace=True)
    df.to_csv(feature_importance_path)
    print(f"✅ 特征重要性数据已保存至: {feature_importance_path}")

    # 12. 可视化
    # 选择中心像素的输入值（只取前20个波段）
    center_inputs = normalized_img[:, :, labels.shape[0] // 2, labels.shape[1] // 2]  # 使用标签图的中心

    # 分别保存两张图表
    input_plot_path = os.path.join(save_dir, "original_inputs_plot.png")
    plot_original_inputs(center_inputs, BAND_NAMES, input_plot_path)

    gradient_plot_path = os.path.join(save_dir, "gradients_plot.png")
    plot_gradients(grads_spatial_avg, BAND_NAMES, gradient_plot_path)

    # 13. 保存原始输入和梯度数据
    np.save(os.path.join(save_dir, "original_inputs.npy"), img_data)
    np.save(os.path.join(save_dir, "normalized_inputs.npy"), normalized_img)
    np.save(os.path.join(save_dir, "gradients.npy"), grads)
    print("✅ 所有数据文件已保存")


# ---------- 入口 ----------
if __name__ == "__main__":
    # 配置路径
    cfg_path = r"C:\Users\Think\Desktop\DeepSatModels-main\configs\PASTIS24\TSViT_fold5.yaml"
    weights_path = r"C:\Users\Think\Desktop\模型\logs\门控自适应8684\best.pth"
    pickle_file_path = r"C:\Users\Think\Desktop\bq\bq_new_new\kuochong_30\64\total2\20369_0_0.pickle"
    save_dir = r"C:\Users\Think\Desktop\feature_importance_results"
    device_ids = [0]  # 使用 GPU 0

    # 添加CUDA初始化检查
    torch.cuda.init()
    if not torch.cuda.is_initialized():
        print("⚠️ CUDA未正确初始化，尝试使用CPU")
        device_ids = []  # 回退到CPU

    main(
        cfg_path=cfg_path,
        weights_path=weights_path,
        device_ids=device_ids,
        save_dir=save_dir,
        pickle_file_path=pickle_file_path
    )