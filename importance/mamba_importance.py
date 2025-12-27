#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
门控权重分析脚本 - 支持时间和空间两种模式
分析注意力分支和Mamba分支的门控权重分布
分析Mamba层的位置关注度（时间步或空间位置）
"""

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

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入自定义模块
from data import get_dataloaders
from data.PASTIS24.data_transforms import Normalize
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint

# ===========================================
# 直接设置路径和参数
# ===========================================

CFG_PATH = r"C:\Users\Think\Desktop\DeepSatModels-main\configs\PASTIS24\TSViT_fold5.yaml"
WEIGHTS_PATH = r"C:\Users\Think\Desktop\模型\logs\门控自适应8684\best.pth"
SAVE_DIR = r"C:\Users\Think\Desktop\gate_analysis"
PICKLE_FILE = r"C:\Users\Think\Desktop\bq\bq_new_new\kuochong_30\64\total2\20369_1_0.pickle"
NUM_SAMPLES = 5
DEVICE_IDS = [0]
ANALYSIS_MODE = 'spatial'  # 可选 'temporal' 或 'spatial'


# ===========================================
# 辅助函数
# ===========================================


def custom_normalize(data, mean, std):
    """手动应用归一化处理，保持时间步长不变"""
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
    """
    准备符合模型输入的张量
    """
    doy_normalized = doys / 365.0
    doy_channel = doy_normalized[:, np.newaxis, np.newaxis, np.newaxis]
    doy_channel = np.broadcast_to(
        doy_channel,
        (doy_normalized.shape[0], 1, normalized_img.shape[2], normalized_img.shape[3])
    )

    model_input = np.concatenate([normalized_img, doy_channel], axis=1)
    return model_input.astype(np.float32)


def process_time_features(xt, device):
    """处理时间特征，避免索引错误"""
    xt = torch.clamp(xt * 365.0001, 0, 365)
    xt = xt.to(torch.int64)
    max_val = xt.max().item()
    if max_val >= 366:
        print(f"⚠️ 警告: 最大时间特征值 {max_val} 超过365，将被裁剪")
        xt = torch.clamp(xt, 0, 365)
    xt = F.one_hot(xt, num_classes=366).to(torch.float32)
    return xt


# ===========================================
# 门控权重可视化函数
# ===========================================

def plot_gate_weights_line(gate_data, save_path):
    """
    绘制每个块中注意力分支和Mamba分支的平均权重折线图
    :param gate_data: 门控权重数据 [block_idx, attn_mean, mamba_mean]
    :param save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 8))

    # 提取数据
    block_indices = [d[0] for d in gate_data]
    attn_means = [d[1] for d in gate_data]
    mamba_means = [d[2] for d in gate_data]

    # 绘制折线图
    plt.plot(block_indices, attn_means, marker='o', linestyle='-', color='blue', label='Attention Branch')
    plt.plot(block_indices, mamba_means, marker='s', linestyle='-', color='green', label='Mamba Branch')

    # 设置图表属性
    plt.title('平均门控权重随块变化趋势', fontsize=16)
    plt.xlabel('块索引', fontsize=12)
    plt.ylabel('平均权重值', fontsize=12)
    plt.xticks(block_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存门控权重折线图: {save_path}")


def plot_gate_weights_box(gate_data, save_path):
    """
    绘制每个块中门控权重的箱线图
    :param gate_data: 门控权重数据 [block_idx, weights]
    :param save_path: 图片保存路径
    """
    plt.figure(figsize=(15, 8))

    # 准备箱线图数据
    data_to_plot = []
    labels = []
    for block_idx, weights in gate_data:
        data_to_plot.append(weights)
        labels.append(f'块 {block_idx}')

    # 绘制箱线图
    plt.boxplot(data_to_plot, labels=labels, showfliers=False)

    # 设置图表属性
    plt.title('门控权重分布随块变化', fontsize=16)
    plt.xlabel('块索引', fontsize=12)
    plt.ylabel('权重值', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存门控权重箱线图: {save_path}")


def plot_single_feature_importance(block_idx, feature_imp, save_dir):
    """为单个块绘制特征关注度热力图"""
    # 如果输入是二维的，计算空间位置的平均值
    if len(feature_imp.shape) > 1:
        print(f"特征关注度数据为二维，形状: {feature_imp.shape}，计算空间位置平均")
        feature_imp = np.mean(feature_imp, axis=0)  # 沿空间位置平均

    print(f"处理后的特征关注度数据形状: {feature_imp.shape}")

    # 应用softmax归一化
    normalized_imp = softmax(feature_imp)

    # 绘制热力图
    plt.figure(figsize=(15, 4))
    sns.heatmap(
        normalized_imp.reshape(1, -1),  # 确保是二维数据
        cmap='viridis',
        cbar=True,
        annot=False,
        yticklabels=False
    )
    plt.title(f'块 {block_idx} 特征关注度')
    plt.xlabel('特征维度')

    # 保存图像
    save_path = os.path.join(save_dir, f"feature_importance_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存特征关注度图(块 {block_idx}): {save_path}")


def plot_single_timestep_importance(block_idx, timestep_imp, save_dir):
    """为单个块绘制时间步关注度热力图"""
    print(f"时间步关注度数据形状: {timestep_imp.shape}")

    # 如果输入是二维的，计算空间位置的平均值
    if len(timestep_imp.shape) > 1:
        print(f"时间步关注度数据为二维，形状: {timestep_imp.shape}，计算空间位置平均")
        timestep_imp = np.mean(timestep_imp, axis=0)  # 沿空间位置平均

    print(f"处理后的时间步关注度数据形状: {timestep_imp.shape}")

    # 应用softmax归一化
    normalized_imp = softmax(timestep_imp)
    print(f"normalized_imp shape after softmax: {normalized_imp.shape}")

    # ==============================
    # 绘制热力图
    # ==============================
    plt.figure(figsize=(15, 4))
    sns.heatmap(
        normalized_imp.reshape(1, -1),  # 确保是二维数据
        cmap='magma',
        cbar=True,
        annot=False,
        yticklabels=False
    )
    plt.title(f'块 {block_idx} 时间步关注度')
    plt.xlabel('时间步')

    # 保存图像
    save_path = os.path.join(save_dir, f"timestep_importance_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存时间步关注度图(块 {block_idx}): {save_path}")


def plot_single_feature_line(block_idx, feature_imp, save_dir):
    """为单个块绘制特征关注度折线图"""
    # 如果输入是二维的，计算空间位置的平均值
    if len(feature_imp.shape) > 1:
        print(f"特征关注度数据为二维，形状: {feature_imp.shape}，计算空间位置平均")
        feature_imp = np.mean(feature_imp, axis=0)  # 沿空间位置平均

    print(f"处理后的特征关注度数据形状: {feature_imp.shape}")

    plt.figure(figsize=(15, 6))

    # 应用softmax归一化
    normalized_imp = softmax(feature_imp)

    # 绘制折线图
    plt.plot(normalized_imp, marker='o', linestyle='-', color='blue')

    # 设置图表属性
    plt.title(f'块 {block_idx} 特征关注度分布', fontsize=16)
    plt.xlabel('特征维度索引', fontsize=12)
    plt.ylabel('关注度值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    save_path = os.path.join(save_dir, f"feature_importance_line_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存特征关注度折线图(块 {block_idx}): {save_path}")


def plot_single_timestep_line(block_idx, timestep_imp, save_dir):
    """为单个块绘制时间步关注度折线图并保存CSV数据"""
    print(f"timestep_imp shape before processing: {timestep_imp.shape}")

    # 如果输入是二维的，计算空间位置的平均值
    if len(timestep_imp.shape) > 1:
        print(f"时间步关注度数据为二维，形状: {timestep_imp.shape}，计算空间位置平均")
        timestep_imp = np.mean(timestep_imp, axis=0)  # 沿空间位置平均

    print(f"处理后的时间步关注度数据形状: {timestep_imp.shape}")

    # 应用softmax归一化
    normalized_imp = softmax(timestep_imp)
    print(f"normalized_imp shape after softmax: {normalized_imp.shape}")

    # ==============================
    # 保存CSV数据 (折线图使用的归一化数据)
    # ==============================
    csv_path = os.path.join(save_dir, f"timestep_importance_block_{block_idx}.csv")
    df = pd.DataFrame({
        'timestep_index': range(len(normalized_imp)),
        'normalized_importance': normalized_imp
    })
    df.to_csv(csv_path, index=False)
    print(f"✅ 保存时间步关注度折线图数据(块 {block_idx}): {csv_path}")
    print(f"📊 时间步数量: {len(normalized_imp)}")
    print(f"📈 归一化数据范围: {normalized_imp.min():.6f} - {normalized_imp.max():.6f}")
    print(f"∑ 概率总和: {normalized_imp.sum():.6f}")

    # ==============================
    # 绘制折线图
    # ==============================
    plt.figure(figsize=(15, 6))
    plt.plot(normalized_imp, marker='s', linestyle='-', color='green')
    plt.title(f'块 {block_idx} 时间步关注度分布', fontsize=16)
    plt.xlabel('时间步索引', fontsize=12)
    plt.ylabel('关注度值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    save_path = os.path.join(save_dir, f"timestep_importance_line_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存时间步关注度折线图(块 {block_idx}): {save_path}")


# ===========================================
# 空间位置关注度可视化函数
# ===========================================

def plot_single_space_importance(block_idx, space_imp, save_dir, grid_size, patch_size, image_size):
    """
    为单个块绘制空间位置关注度热力图（块级别）
    :param block_idx: 块索引
    :param space_imp: 空间位置重要性数据 (num_patches,)
    :param save_dir: 保存目录
    :param grid_size: 网格大小（每个维度的块数）
    :param patch_size: 每个块的像素大小
    :param image_size: 原始图像尺寸
    """
    print(f"空间位置关注度数据形状: {space_imp.shape}")

    # 确保空间位置数量匹配网格大小
    expected_size = grid_size * grid_size
    if space_imp.size != expected_size:
        print(f"⚠️ 警告: 空间位置数据大小 {space_imp.size} 与预期网格大小 {expected_size} 不匹配")
        return None

    # 应用softmax归一化
    normalized_imp =space_imp

    # 重塑为二维网格
    grid_imp = normalized_imp.reshape(grid_size, grid_size)

    # 绘制热力图（块级别）
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

    # 保存图像
    save_path = os.path.join(save_dir, f"space_importance_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存空间位置关注度图(块 {block_idx}): {save_path}")

    # 保存CSV数据
    csv_path = os.path.join(save_dir, f"space_importance_block_{block_idx}.csv")
    np.savetxt(csv_path, grid_imp, delimiter=",")
    print(f"✅ 保存空间位置关注度数据(块 {block_idx}): {csv_path}")

    return grid_imp


def plot_pixel_importance(block_idx, grid_imp, save_dir, grid_size, patch_size, image_size):
    """
    为单个块绘制像素级空间位置关注度热力图
    :param block_idx: 块索引
    :param grid_imp: 网格级重要性数据 (grid_size, grid_size)
    :param save_dir: 保存目录
    :param grid_size: 网格大小（每个维度的块数）
    :param patch_size: 每个块的像素大小
    :param image_size: 原始图像尺寸
    """
    # 创建全尺寸的重要性图
    pixel_imp = np.zeros((image_size, image_size))

    # 计算每个块对应的像素区域
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算当前块在原始图像中的像素范围
            start_h = i * patch_size
            end_h = min((i + 1) * patch_size, image_size)
            start_w = j * patch_size
            end_w = min((j + 1) * patch_size, image_size)

            # 将块的重要性值赋给对应像素区域
            pixel_imp[start_h:end_h, start_w:end_w] = grid_imp[i, j]

    # ================================
    # 绘制像素级热力图
    # ================================
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

    # 保存图像
    save_path = os.path.join(save_dir, f"pixel_importance_block_{block_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存像素级空间位置关注度图(块 {block_idx}): {save_path}")

    # 保存CSV数据
    csv_path = os.path.join(save_dir, f"pixel_importance_block_{block_idx}.csv")
    np.savetxt(csv_path, pixel_imp, delimiter=",")
    print(f"✅ 保存像素级空间位置关注度数据(块 {block_idx}): {csv_path}")

    # ================================
    # 识别关键像素区域
    # ================================
    # 找出重要性最高的像素区域
    max_imp = np.max(pixel_imp)
    threshold = max_imp * 0.7  # 70%阈值
    high_imp_coords = np.argwhere(pixel_imp > threshold)

    # 分析关键区域
    if len(high_imp_coords) > 0:
        print(f"🔍 块 {block_idx} 关键像素区域分析:")
        print(f"  - 高关注度像素数量: {len(high_imp_coords)}")
        print(f"  - 最大关注度值: {max_imp:.4f}")
        print(f"  - 高关注度区域边界:")
        min_h, min_w = np.min(high_imp_coords, axis=0)
        max_h, max_w = np.max(high_imp_coords, axis=0)
        print(f"    X: {min_w}-{max_w}, Y: {min_h}-{max_h}")
        print(f"    Width: {max_w - min_w}px, Height: {max_h - min_h}px")

    return pixel_imp


def plot_space_position_importance_line(space_data, save_path):
    """
    绘制空间位置平均关注度随块变化的折线图
    :param space_data: 空间位置数据 [(block_idx, avg_importance), ...]
    :param save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 8))

    # 提取数据
    block_indices = [d[0] for d in space_data]
    avg_importance = [d[1] for d in space_data]

    # 绘制折线图
    plt.plot(block_indices, avg_importance, marker='o', linestyle='-', color='purple', label='平均关注度')

    # 设置图表属性
    plt.title('空间位置平均关注度随块变化趋势', fontsize=16)
    plt.xlabel('块索引', fontsize=12)
    plt.ylabel('平均关注度值', fontsize=12)
    plt.xticks(block_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存空间位置关注度折线图: {save_path}")


# ===========================================
# 修改后的前向传播函数（捕获门控权重）
# ===========================================

def modified_forward_with_gate_weights(inputs, net, analysis_mode):
    """
    修改后的模型前向传播函数，捕获门控权重
    返回：
    - logits: 模型输出
    - gate_weights_data: 门控权重数据 [block_idx, weights]
    """
    # 提取输入形状
    B, T, C, H, W = inputs.shape
    inputs = inputs.float()

    # 设置分析模式
    net.set_analysis_mode(analysis_mode)
    print(f"✅ 设置分析模式: {analysis_mode}")

    # 提取时间特征
    xt = inputs[:, :, -1, 0, 0]
    xt = process_time_features(xt, inputs.device)
    xt = xt.reshape(-1, 366)

    # 应用时间位置嵌入
    temporal_pos_embedding = net.to_temporal_embedding_input(xt).reshape(B, T, net.dim)

    # 准备patch嵌入
    x = inputs[:, :, :-1]  # 移除时间特征通道

    # 确保空间维度能被patch_size整除
    assert H % net.patch_size == 0, f"高度 {H} 不能被 patch_size {net.patch_size} 整除"
    assert W % net.patch_size == 0, f"宽度 {W} 不能被 patch_size {net.patch_size} 整除"

    # 计算patch数量
    num_patches_h = H // net.patch_size
    num_patches_w = W // net.patch_size
    num_patches = num_patches_h * num_patches_w

    # 计算网格大小（用于空间分析）
    grid_size = num_patches_h

    # 手动实现重排操作
    x = x.unfold(3, net.patch_size, net.patch_size)
    x = x.unfold(4, net.patch_size, net.patch_size)
    x = x.permute(0, 3, 4, 1, 2, 5, 6)
    x = x.reshape(B * num_patches_h * num_patches_w, T, 20 * net.patch_size * net.patch_size)

    # 应用线性变换
    x = net.to_patch_embedding[1](x)

    # 添加时间位置嵌入
    x = x.reshape(B, num_patches, T, net.dim)
    x += temporal_pos_embedding.unsqueeze(1)
    x = x.reshape(B * num_patches, T, net.dim)

    # 添加时间token
    cls_temporal_tokens = net.temporal_token.repeat(B * num_patches, 1, 1)
    x = torch.cat((cls_temporal_tokens, x), dim=1)

    # 存储门控权重数据
    gate_weights_data = []

    # 存储Mamba特征和时间步关注度
    feature_importance_data = []
    timestep_importance_data = []

    # 存储位置重要性数据
    position_importance_data = []

    # 启用门控权重收集
    for block in net.temporal_transformer.layers:
        # 重置Mamba分析属性
        block.feature_importance = None
        block.timestep_importance = None
        block.attention_scores = None

    # 前向传播时间变换器
    x = net.temporal_transformer(x)

    # 收集门控权重数据和Mamba分析数据
    for block_idx, block in enumerate(net.temporal_transformer.layers):
        # 收集门控权重
        if hasattr(block, 'attn_weights') and block.attn_weights:
            # 取最后一次记录的权重（当前批次）
            # 注意：这里已经是NumPy数组，不需要detach()
            attn_weights_np = block.attn_weights[-1].flatten()
            gate_weights_data.append((block_idx, attn_weights_np))

        # 收集Mamba特征关注度
        if hasattr(block, 'feature_importance') and block.feature_importance is not None:
            # 转换为NumPy数组
            feat_imp = block.feature_importance.detach().cpu().numpy()
            feature_importance_data.append((block_idx, feat_imp))

        # 收集Mamba时间步关注度
        if hasattr(block, 'timestep_importance') and block.timestep_importance is not None:
            # 转换为NumPy数组
            timestep_imp = block.timestep_importance.detach().cpu().numpy()
            # 去掉前4个类别token (只保留时间步关注度)
            if timestep_imp.shape[1] > 4:  # 确保有类别token
                timestep_imp = timestep_imp[:, 4:]  # 去掉前4个类别token
            timestep_importance_data.append((block_idx, timestep_imp))

    # 收集位置重要性数据（时间或空间）
    if analysis_mode == 'temporal':
        # 时间位置重要性
        temporal_pos_imp = net.get_temporal_position_importance()
        if temporal_pos_imp:
            print(f"✅ 检测到时间位置重要性数据: {len(temporal_pos_imp)} 个块")
            for block_idx, imp in enumerate(temporal_pos_imp):
                # 确保是NumPy数组
                if isinstance(imp, torch.Tensor):
                    imp = imp.detach().cpu().numpy()
                position_importance_data.append((block_idx, imp))
                print(f"✅ 收集到时间块 {block_idx} 的位置重要性, 形状: {imp.shape}")
        else:
            print("⚠️ 未检测到时间位置重要性数据")

    # 空间变换器
    x = x[:, :net.num_classes]
    x = x.reshape(B, num_patches, net.num_classes, net.dim)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B * net.num_classes, num_patches, net.dim)

    # 确保空间位置嵌入大小匹配
    space_pos_embedding = net.space_pos_embedding[:, :num_patches] if net.space_pos_embedding.shape[
                                                                          1] > num_patches else net.space_pos_embedding
    x += space_pos_embedding

    # 应用dropout
    if hasattr(net, 'dropout'):
        x = net.dropout(x)

    # 前向传播空间变换器
    x = net.space_transformer(x)

    # 关键修改：在空间变换器前向传播后收集空间位置重要性数据
    if analysis_mode == 'spatial' and hasattr(net.space_transformer, 'get_space_position_importance'):
        space_pos_imp = net.space_transformer.get_space_position_importance()
        if space_pos_imp:
            print(f"✅ 捕获空间位置重要性数据: {len(space_pos_imp)} 个块")
            for block_idx, imp in enumerate(space_pos_imp):
                # 确保是NumPy数组
                if isinstance(imp, torch.Tensor):
                    imp = imp.detach().cpu().numpy()
                position_importance_data.append((block_idx, imp))
                print(f"✅ 收集到空间块 {block_idx} 的位置重要性, 形状: {imp.shape}")
        else:
            print("⚠️ 空间位置重要性数据为空")

    # MLP头部
    x = net.mlp_head(x.reshape(-1, net.dim))

    # 重塑输出
    x = x.reshape(B, net.num_classes, num_patches, net.patch_size ** 2)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(B, num_patches_h, num_patches_w, net.patch_size, net.patch_size, net.num_classes)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, num_patches_h * net.patch_size, num_patches_w * net.patch_size, net.num_classes)
    x = x.permute(0, 3, 1, 2)

    return x, gate_weights_data, feature_importance_data, timestep_importance_data, position_importance_data, grid_size


# ===========================================
# 计算并保存空间patch块重要性
# ===========================================

def save_spatial_patch_importance(position_importance_data, save_dir, num_patches):
    """
    为每一层Transformer计算并保存空间patch块的重要性向量
    :param position_importance_data: 位置重要性数据列表 [(block_idx, importance_vector), ...]
    :param save_dir: 保存目录
    :param num_patches: 空间patch块数量
    """
    # 创建汇总DataFrame
    summary_df = pd.DataFrame()

    # 为每一层保存单独的文件
    for block_idx, imp_vec in position_importance_data:
        # 确保向量长度正确
        if len(imp_vec) < num_patches:
            # 用0填充不足部分
            padded_vec = np.zeros(num_patches)
            padded_vec[:len(imp_vec)] = imp_vec
            imp_vec = padded_vec
            print(f"⚠️ 块 {block_idx} 重要性向量长度不足 {num_patches}，已用0填充")
        elif len(imp_vec) > num_patches:
            # 截断超过部分
            imp_vec = imp_vec[:num_patches]
            print(f"⚠️ 块 {block_idx} 重要性向量长度超过 {num_patches}，已截断")

        # 创建DataFrame保存当前块的重要性
        df = pd.DataFrame({
            'patch_index': range(num_patches),
            'importance': imp_vec
        })

        # 保存当前块的CSV
        csv_path = os.path.join(save_dir, f"spatial_patch_importance_block_{block_idx}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ 保存块 {block_idx} 的空间patch重要性: {csv_path}")

        # 添加到汇总DataFrame
        summary_df[f'block_{block_idx}'] = imp_vec

    # 保存汇总CSV
    if not summary_df.empty:
        summary_df.insert(0, 'patch_index', range(num_patches))
        summary_csv_path = os.path.join(save_dir, "spatial_patch_importance_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"✅ 保存空间patch重要性汇总: {summary_csv_path}")
    else:
        print("⚠️ 未生成空间patch重要性汇总文件，无有效数据")


# ===========================================
# 主函数
# ===========================================

def main():
    global ANALYSIS_MODE

    print("=" * 50)
    print(f"模型配置文件: {CFG_PATH}")
    print(f"模型权重文件: {WEIGHTS_PATH}")
    print(f"结果保存目录: {SAVE_DIR}")
    print(f"使用设备: {'GPU' if DEVICE_IDS else 'CPU'} {DEVICE_IDS}")
    print(f"分析模式: {ANALYSIS_MODE}")
    print("=" * 50)

    # 0. 设备设置
    device = get_device(DEVICE_IDS, allow_cpu=True)

    # 1. 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 根据分析模式创建主目录
    if ANALYSIS_MODE == 'temporal':
        main_dir = os.path.join(SAVE_DIR, "temporal_analysis")
    elif ANALYSIS_MODE == 'spatial':
        main_dir = os.path.join(SAVE_DIR, "spatial_analysis")
    else:
        raise ValueError(f"无效的分析模式: {ANALYSIS_MODE}")

    os.makedirs(main_dir, exist_ok=True)
    print(f"📁 主分析目录: {main_dir}")

    # 创建子目录
    gate_dir = os.path.join(main_dir, "gate_weights_analysis")
    feature_dir = os.path.join(main_dir, "feature_importance")
    timestep_dir = os.path.join(main_dir, "timestep_importance")
    position_dir = os.path.join(main_dir, "position_importance")
    patch_importance_dir = os.path.join(main_dir, "patch_importance")  # 新增目录

    # 确保所有目录都存在
    os.makedirs(gate_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(timestep_dir, exist_ok=True)
    os.makedirs(position_dir, exist_ok=True)
    os.makedirs(patch_importance_dir, exist_ok=True)  # 确保新目录存在

    # 2. 读取配置
    config = read_yaml(CFG_PATH)
    config["local_device_ids"] = DEVICE_IDS

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
    print("🔧 构建并加载模型...")
    net = get_model(config, device)
    load_from_checkpoint(net, WEIGHTS_PATH, device)
    net.to(device).eval()

    # 获取模型参数
    patch_size = getattr(net, 'patch_size', 16)
    print(f"ℹ️ 使用patch_size: {patch_size}")

    # 6. 加载样本数据
    if PICKLE_FILE and os.path.exists(PICKLE_FILE):
        print(f"🚀 加载样本: {PICKLE_FILE}")
        with open(PICKLE_FILE, 'rb') as f:
            data = pickle.load(f)
        img_data, labels, doys = data['img'], data['labels'], data['doy']

        # 打印原始形状信息
        print(
            f"📊 原始数据形状 - 时间步: {img_data.shape[0]}, 通道: {img_data.shape[1]}, 空间: {img_data.shape[2]}x{img_data.shape[3]}")

        # 应用自定义归一化
        normalized_img = custom_normalize(img_data, mean, std)

        # 准备模型输入
        model_input = prepare_model_input(normalized_img, doys)

        # 打印调整后的形状
        T, C, H, W = model_input.shape
        print(f"🔄 模型输入形状 - 时间步: {T}, 通道: {C}, 空间: {H}x{W}")
        print(f"ℹ️ 空间维度 {H}x{W} 应能被 {patch_size} 整除: {H % patch_size == 0 and W % patch_size == 0}")

        # 计算空间patch块数量
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w
        print(f"ℹ️ 空间patch块数量: {num_patches} ({num_patches_h}x{num_patches_w})")

        # 转换为张量并调整维度顺序
        inputs = torch.tensor(model_input, dtype=torch.float32)
        inputs = inputs.unsqueeze(0)  # 添加批次维度 [1, T, C, H, W]
        inputs = inputs.to(device)
        print(f"📦 输入张量形状: {inputs.shape}")

        # 运行模型前向传播 - 捕获门控权重和Mamba分析数据
        print("运行模型前向传播并捕获分析数据...")
        with torch.no_grad():
            logits, gate_weights_data, feature_importance_data, timestep_importance_data, position_importance_data, grid_size = \
                modified_forward_with_gate_weights(inputs, net, ANALYSIS_MODE)

        # 处理门控权重数据
        if gate_weights_data:
            print(f"✅ 检测到 {len(gate_weights_data)} 个块的门控权重")

            # 准备折线图数据
            line_plot_data = []
            for block_idx, weights in gate_weights_data:
                attn_mean = np.mean(weights)  # 注意力分支平均权重
                mamba_mean = 1 - attn_mean  # Mamba分支平均权重
                line_plot_data.append((block_idx, attn_mean, mamba_mean))

            # 绘制折线图
            line_path = os.path.join(gate_dir, "gate_weights_line.png")
            plot_gate_weights_line(line_plot_data, line_path)

            # 绘制箱线图
            box_path = os.path.join(gate_dir, "gate_weights_box.png")
            plot_gate_weights_box(gate_weights_data, box_path)

            # 保存CSV数据
            for block_idx, weights in gate_weights_data:
                print(f"Saving gate weights for block {block_idx}, weights shape: {weights.shape}")
                csv_path = os.path.join(gate_dir, f"gate_weights_block_{block_idx}.csv")
                pd.DataFrame(weights, columns=['weight']).to_csv(csv_path, index=False)
        else:
            print("⚠️ 未检测到门控权重数据，请检查模型结构")

        # 处理Mamba特征关注度数据
        if feature_importance_data:
            print(f"✅ 检测到 {len(feature_importance_data)} 个块的特征关注度")

            # 为每个块单独绘制特征关注度图
            for block_idx, feat_imp in feature_importance_data:
                # 绘制热力图
                plot_single_feature_importance(block_idx, feat_imp, feature_dir)

                # 绘制折线图
                plot_single_feature_line(block_idx, feat_imp, feature_dir)

                # 保存CSV数据
                csv_path = os.path.join(feature_dir, f"feature_importance_block_{block_idx}.csv")
                pd.DataFrame(feat_imp).to_csv(csv_path, index=False)
        else:
            print("⚠️ 未检测到Mamba特征关注度数据")

        # 处理Mamba时间步关注度数据
        if timestep_importance_data:
            print(f"✅ 检测到 {len(timestep_importance_data)} 个块的时间步关注度")

            # 为每个块单独绘制时间步关注度图
            for block_idx, time_imp in timestep_importance_data:
                # 绘制热力图
                plot_single_timestep_importance(block_idx, time_imp, timestep_dir)

                # 绘制折线图并保存CSV
                plot_single_timestep_line(block_idx, time_imp, timestep_dir)
        else:
            print("⚠️ 未检测到Mamba时间步关注度数据")

        # 处理位置重要性数据
        if position_importance_data:
            print(f"✅ 检测到 {len(position_importance_data)} 个块的位置重要性")

            # 时间模式：绘制时间步关注度折线图
            if ANALYSIS_MODE == 'temporal':
                for block_idx, pos_imp in position_importance_data:
                    plot_single_timestep_line(block_idx, pos_imp, position_dir)

            # 空间模式：处理空间位置重要性
            elif ANALYSIS_MODE == 'spatial':
                # 保存空间patch块重要性
                save_spatial_patch_importance(
                    position_importance_data=position_importance_data,
                    save_dir=patch_importance_dir,
                    num_patches=num_patches
                )

                # 绘制空间位置热力图
                line_plot_data = []
                for block_idx, pos_imp in position_importance_data:
                    # 绘制空间位置热力图（块级别）
                    grid_imp = plot_single_space_importance(
                        block_idx=block_idx,
                        space_imp=pos_imp,
                        save_dir=position_dir,
                        grid_size=grid_size,
                        patch_size=patch_size,
                        image_size=H  # 使用原始图像高度
                    )

                    # 绘制像素级热力图
                    if grid_imp is not None:
                        pixel_imp = plot_pixel_importance(
                            block_idx=block_idx,
                            grid_imp=grid_imp,
                            save_dir=position_dir,
                            grid_size=grid_size,
                            patch_size=patch_size,
                            image_size=H  # 使用原始图像高度
                        )

                    # 计算平均关注度用于折线图
                    avg_imp = np.mean(pos_imp)
                    line_plot_data.append((block_idx, avg_imp))

                # 绘制空间位置平均关注度折线图
                if line_plot_data:
                    line_path = os.path.join(position_dir, "space_importance_trend.png")
                    plot_space_position_importance_line(line_plot_data, line_path)
        else:
            print("⚠️ 未检测到位置重要性数据")
    else:
        print(f"⚠️ 未找到样本文件 {PICKLE_FILE}，请检查路径")

    print("\n✅ 分析完成！结果保存在:", main_dir)
    print("=" * 50)


if __name__ == "__main__":
    main()