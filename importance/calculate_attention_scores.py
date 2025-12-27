#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
注意力分支定性分析脚本（修改版）
主要修改：
1. 添加门控权重CSV保存功能
2. 添加门控权重可视化功能
3. 支持空间模式下的门控权重分析
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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
import matplotlib as mpl

# 设置全局绘图样式
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
SAVE_DIR = r"C:\Users\Think\Desktop\attention_analysis"
PICKLE_FILE = r"C:\Users\Think\Desktop\bq\bq_new_new\kuochong_30\64\total2\20369_1_0.pickle"
NUM_SAMPLES = 1
DEVICE_IDS = [0]
ANALYSIS_MODE = 'spatial'  # 可选 'temporal' 或 'spatial'


# ===========================================
# 辅助函数
# ===========================================

def save_gate_weights(weights_data, save_dir, prefix, num_positions):
    """保存门控权重到CSV文件"""
    os.makedirs(save_dir, exist_ok=True)

    # 确保权重数据不为空
    if not weights_data:
        print("⚠️ 警告: 权重数据为空，无法保存CSV")
        return

    for block_idx, block_data in enumerate(weights_data):
        # 检查数据有效性
        if 'attn' not in block_data or 'mamba' not in block_data:
            print(f"⚠️ 块 {block_idx} 缺少权重数据，跳过保存")
            continue

        # 获取当前块的权重数据
        attn_weights = block_data['attn']
        mamba_weights = block_data['mamba']

        # 确保是张量
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.numpy()
        if isinstance(mamba_weights, torch.Tensor):
            mamba_weights = mamba_weights.numpy()

        # 检查形状
        if attn_weights.ndim > 1:
            # 取第一个样本
            attn_weights = attn_weights[0]
        if mamba_weights.ndim > 1:
            mamba_weights = mamba_weights[0]

        # 确保长度正确
        if len(attn_weights) < num_positions:
            # 填充零值
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

        # 创建DataFrame
        df = pd.DataFrame({
            'position': range(num_positions),
            'attn_weight': attn_weights,
            'mamba_weight': mamba_weights
        })

        # 保存CSV
        csv_path = os.path.join(save_dir, f"{prefix}_gate_weights_block_{block_idx}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ 保存门控权重CSV: {csv_path}")


def plot_gate_weights(weights_data, save_dir, prefix, num_positions, figsize=(10, 6)):
    """
    绘制门控权重柱状图（如您提供的图片样式）
    :param weights_data: 门控权重数据
    :param save_dir: 保存目录
    :param prefix: 文件名前缀
    :param num_positions: 位置数量
    :param figsize: 图片尺寸
    """
    os.makedirs(save_dir, exist_ok=True)

    # 确保权重数据不为空
    if not weights_data:
        print("⚠️ 警告: 权重数据为空，无法绘制图表")
        return

    for block_idx, block_data in enumerate(weights_data):
        # 检查数据有效性
        if 'attn' not in block_data or 'mamba' not in block_data:
            print(f"⚠️ 块 {block_idx} 缺少权重数据，跳过绘图")
            continue

        # 提取当前块的权重
        attn_weights = block_data['attn']
        mamba_weights = block_data['mamba']

        # 确保是张量
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.numpy()
        if isinstance(mamba_weights, torch.Tensor):
            mamba_weights = mamba_weights.numpy()

        # 检查形状
        if attn_weights.ndim > 1:
            # 取第一个样本
            attn_weights = attn_weights[0]
        if mamba_weights.ndim > 1:
            mamba_weights = mamba_weights[0]

        # 确保长度正确
        if len(attn_weights) > num_positions:
            attn_weights = attn_weights[:num_positions]
        if len(mamba_weights) > num_positions:
            mamba_weights = mamba_weights[:num_positions]

        # 创建专业学术图表
        fig, ax = plt.subplots(figsize=figsize, dpi=300)

        # 设置专业配色
        attention_color = '#4e79a7'  # 深蓝色
        mamba_color = '#f28e2b'  # 橙色

        # 绘制柱状图
        positions = np.arange(num_positions)
        bar_width = 0.4

        # 绘制注意力权重
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

        # 绘制Mamba权重
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

        # 添加数值标签
        for i, (attn_w, mamba_w) in enumerate(zip(attn_weights, mamba_weights)):
            ax.text(
                i, attn_w + 0.005, f"{attn_w:.3f}",
                ha='center', fontsize=8, fontweight='bold'
            )
            ax.text(
                i + bar_width, mamba_w + 0.005, f"{mamba_w:.3f}",
                ha='center', fontsize=8, fontweight='bold'
            )

        # 添加标题和标签
        ax.set_title(f'Gate Weights - Block {block_idx + 1}', fontsize=14, pad=15)
        ax.set_xlabel('Patch Position', fontsize=12, labelpad=10)
        ax.set_ylabel('Gate Weight Value', fontsize=12, labelpad=10)
        ax.set_xticks(positions + bar_width / 2)
        ax.set_xticklabels([f'{i}' for i in range(num_positions)], fontsize=10)

        # 设置Y轴范围
        max_val = max(np.max(attn_weights), np.max(mamba_weights))
        ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1.0)

        # 添加图例和网格
        ax.legend(fontsize=10, frameon=True, shadow=True, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        img_path = os.path.join(save_dir, f"{prefix}_gate_weights_block_{block_idx + 1}.png")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存门控权重图: {img_path}")


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
# 二部图可视化函数（修改版）
# ===========================================

def plot_attention_bipartite(attention_matrix, save_path, block_idx, figsize=(12, 6)):
    """
    绘制修改后的二部图（垂直布局），虚化不重要的线并突出重要节点
    :param attention_matrix: 注意力矩阵 [query_len, key_len]
    :param save_path: 图片保存路径
    :param block_idx: 当前块索引
    :param figsize: 图像大小
    """
    T = attention_matrix.shape[0]  # 时间步数量

    # 创建画布
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # 1. 计算节点重要性（基于注意力权重）
    # 输入节点（Key）重要性 = 每列的和（所有输出节点对其的关注度）
    key_importance = attention_matrix.sum(axis=0)
    # 输出节点（Query）重要性 = 每行的和（该节点对所有输入节点的关注度）
    query_importance = attention_matrix.sum(axis=1)

    # 归一化重要性分数（用于节点大小）
    key_importance_norm = (key_importance - key_importance.min()) / (key_importance.max() - key_importance.min() + 1e-8)
    query_importance_norm = (query_importance - query_importance.min()) / (
            query_importance.max() - query_importance.min() + 1e-8)

    # 节点大小范围 [50, 300]
    min_node_size = 50
    max_node_size = 300
    key_sizes = min_node_size + (max_node_size - min_node_size) * key_importance_norm
    query_sizes = min_node_size + (max_node_size - min_node_size) * query_importance_norm

    # 2. 确定重要节点（重要性大于平均值）
    important_key_indices = np.where(key_importance > key_importance.mean())[0]
    important_query_indices = np.where(query_importance > query_importance.mean())[0]

    print(f"重要输入节点: {important_key_indices}")
    print(f"重要输出节点: {important_query_indices}")

    # 3. 节点位置计算（垂直布局）
    x_pos_top = np.linspace(0, 1, T)  # 顶部节点x坐标（输入节点）
    x_pos_bottom = np.linspace(0, 1, T)  # 底部节点x坐标（输出节点）

    # 4. 绘制所有节点
    # 普通节点（灰色）
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

    # 突出重要节点（彩色）
    if len(important_key_indices) > 0:
        ax.scatter(
            x_pos_top[important_key_indices], [1] * len(important_key_indices),
            s=key_sizes[important_key_indices],
            c='skyblue',
            edgecolors='k',
            alpha=1.0,
            zorder=10  # 确保重要节点在最上层
        )

    if len(important_query_indices) > 0:
        ax.scatter(
            x_pos_bottom[important_query_indices], [0] * len(important_query_indices),
            s=query_sizes[important_query_indices],
            c='lightblue',
            edgecolors='k',
            alpha=1.0,
            zorder=10  # 确保重要节点在最上层
        )

    # 5. 添加节点标签（只标记重要节点）
    # for i in range(T):
    #     # 输入节点（顶部）
    #     if i in important_key_indices:
    #         ax.text(
    #             x_pos_top[i], 1.05, '',
    #             ha='center', va='bottom', fontsize=10, fontweight='bold',
    #             bbox=dict(facecolor='skyblue', alpha=0.8, pad=2, edgecolor='k')
    #         )
    #     # 输出节点（底部）
    #     if i in important_query_indices:
    #         ax.text(
    #             x_pos_bottom[i], -0.05, '',
    #             ha='center', va='top', fontsize=10, fontweight='bold',
    #             bbox=dict(facecolor='skyblue', alpha=0.8, pad=2, edgecolor='k')
    #         )

    # 6. 绘制边（用透明度表示权重）
    max_weight = np.max(attention_matrix)
    min_weight = np.min(attention_matrix[attention_matrix > 0])

    # 找出最大注意力位置
    max_attention_idx = np.unravel_index(np.argmax(attention_matrix), attention_matrix.shape)

    # 计算所有边的权重阈值
    weight_threshold = max_weight * 0.05  # 只显示大于最大权重10%的边

    for i in range(T):  # query索引 (输出节点，底部)
        for j in range(T):  # key索引 (输入节点，顶部)
            weight = attention_matrix[i, j]

            # 只绘制显著的边
            if weight > weight_threshold:
                # 计算归一化权重（0-1范围）
                norm_weight = (weight - min_weight) / (max_weight - min_weight + 1e-8)
                linewidth = 0.5 + 2.5 * norm_weight

                # 标记最大注意力边
                is_max = (i, j) == max_attention_idx

                # 判断是否是重要节点之间的连接
                is_important_edge = (j in important_key_indices) and (i in important_query_indices)

                if is_max:
                    # 最大注意力边 - 红色
                    color = 'lightblue'
                    alpha = 0.8
                    linestyle = '-'
                elif is_important_edge:
                    # 重要节点之间的边 - 蓝色
                    color = 'lightblue'
                    alpha = 0.8
                    linestyle = '-'
                else:
                    # 普通边 - 浅灰色，半透明
                    color = 'lightgray'
                    alpha = 0.3
                    linestyle = '--'  # 虚线表示不重要

                # 绘制边（从顶部节点到底部节点）
                ax.plot(
                    [x_pos_top[j], x_pos_bottom[i]],
                    [1, 0],
                    linewidth=linewidth,
                    alpha=alpha,
                    color=color,
                    linestyle=linestyle,
                    zorder=1 if is_important_edge or is_max else 0
                )

    # 7. 添加图例
    # legend_elements = [
    #     Line2D([0], [0], color='skyblue', marker='o', markersize=8, label='重要输入节点', linestyle='None'),
    #     Line2D([0], [0], color='lightgreen', marker='o', markersize=8, label='重要输出节点', linestyle='None'),
    #     Line2D([0], [0], color='lightgray', marker='o', markersize=8, label='普通节点', linestyle='None'),
    #     Line2D([0], [0], color='red', linewidth=2, label='最大注意力边'),
    #     Line2D([0], [0], color='blue', linewidth=2, label='重要节点间连接'),
    #     Line2D([0], [0], color='lightgray', linewidth=1, linestyle='--', label='普通连接')
    # ]

    # ax.legend(
    #     handles=legend_elements,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.1),
    #     ncol=3,
    #     fontsize=9
    # )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.2)
    # ax.set_title(f'注意力二部图 - 块 {block_idx}', fontsize=16, pad=20)
    ax.axis('off')

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 保存二部图: {save_path}")
    return save_path


# ===========================================
# 修改后的前向传播函数（捕获注意力得分）
# ===========================================

def modified_forward_with_attention(inputs, net, mode='temporal'):
    """
    修改后的模型前向传播函数，捕获注意力得分
    返回：
    - logits: 模型输出
    - attention_scores: 各层的注意力得分列表
    """
    # 提取输入形状
    B, T, C, H, W = inputs.shape
    inputs = inputs.float()

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

    # 根据模式设置注意力捕获
    if mode == 'temporal':
        # 时间变换器 - 捕获注意力得分
        for block in net.temporal_transformer.layers:
            if hasattr(block, 'attn'):
                block.attn.return_attention = True  # 启用注意力捕获
        x = net.temporal_transformer(x)

        # 收集时间注意力得分
        attention_scores_list = []
        for block in net.temporal_transformer.layers:
            if hasattr(block, 'attention_scores') and block.attention_scores is not None:
                attention_scores_list.append(block.attention_scores)
    else:
        # 跳过时间注意力捕获
        x = net.temporal_transformer(x)
        attention_scores_list = []

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

    # 根据模式设置空间注意力捕获
    if mode == 'spatial':
        # 设置空间transformer的块以捕获注意力
        for block in net.space_transformer.layers:
            if hasattr(block, 'attn'):
                block.attn.return_attention = True
        x = net.space_transformer(x)

        # 收集空间注意力得分
        attention_scores_list = []
        for block in net.space_transformer.layers:
            if hasattr(block, 'attention_scores') and block.attention_scores is not None:
                attention_scores_list.append(block.attention_scores)
    else:
        # 跳过空间注意力捕获
        x = net.space_transformer(x)

    # MLP头部
    x = net.mlp_head(x.reshape(-1, net.dim))

    # 重塑输出
    x = x.reshape(B, net.num_classes, num_patches, net.patch_size ** 2)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(B, num_patches_h, num_patches_w, net.patch_size, net.patch_size, net.num_classes)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, num_patches_h * net.patch_size, num_patches_w * net.patch_size, net.num_classes)
    x = x.permute(0, 3, 1, 2)
    return x, attention_scores_list


# ===========================================
# 计算每层空间块重要性
# ===========================================

def compute_spatial_block_importance_per_layer(attention_scores_list, num_blocks=16):
    """
    为每一层计算空间块重要性
    :param attention_scores_list: 空间注意力得分列表（每层一个）
    :param num_blocks: 空间块数量（默认16）
    :return: 每层的重要性向量列表
    """
    block_importance_per_layer = []

    for layer_idx, attention_scores in enumerate(attention_scores_list):
        print(f"计算第 {layer_idx + 1} 层空间块重要性...")

        # 处理多头注意力 [batch, heads, query_len, key_len]
        if attention_scores.dim() == 4:
            # 取第一个样本、所有头的平均
            avg_attention = attention_scores.mean(dim=1)[0]  # [query_len, key_len]

            # 计算每个块作为key被关注的程度（列和）
            block_importance = avg_attention.sum(dim=0).detach().cpu().numpy()

            # 确保长度正确
            if len(block_importance) < num_blocks:
                block_importance = np.pad(block_importance, (0, num_blocks - len(block_importance)),
                                          mode='constant', constant_values=0)
            elif len(block_importance) > num_blocks:
                block_importance = block_importance[:num_blocks]

            block_importance_per_layer.append(block_importance)
        else:
            print(f"⚠️ 第 {layer_idx + 1} 层: 不支持的注意力维度 {attention_scores.dim()}")
            # 添加一个NaN向量
            block_importance_per_layer.append(np.full(num_blocks, np.nan))

    return block_importance_per_layer


# ===========================================
# 注意力得分可视化
# ===========================================

def plot_temporal_attention(attention_scores, save_dir, block_idx, input_series=None):
    """处理时间注意力矩阵"""
    print(f"块 {block_idx} 原始时间注意力得分形状: {attention_scores.shape}")

    # 处理多头注意力 [batch, heads, query_len, key_len]
    if attention_scores.dim() == 4:
        # 取第一个样本、所有头的平均
        avg_attention = attention_scores.mean(dim=1)[0]  # [query_len, key_len]
        print(f"平均注意力形状: {avg_attention.shape}")

        # 提取时间步部分 (后21×21)
        num_timesteps = 21
        timestep_attention = avg_attention[-num_timesteps:, -num_timesteps:]
        print(f"时间步注意力形状: {timestep_attention.shape}")

        # 转换为numpy
        timestep_attention = timestep_attention.detach().cpu().numpy()

        # 保存CSV
        csv_path = os.path.join(save_dir, f"timestep_attention_block_{block_idx}.csv")
        pd.DataFrame(timestep_attention).to_csv(csv_path, index=False)

        # 绘制热力图（仅时间步）
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(timestep_attention, annot=False, cmap="viridis", cbar=True, square=True,
                         annot_kws={"size": 16})
        # plt.title(f'时间步注意力热力图 - 块 {block_idx}', fontsize=16)
        plt.xlabel("observation time t_out", fontsize=20)
        plt.ylabel("observation time t_in", fontsize=20)
        # 修改刻度字体大小
        ax.tick_params(axis='x', labelsize=16)  # X轴刻度
        ax.tick_params(axis='y', labelsize=16)  # Y轴刻度

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)  # 颜色条刻度

        img_path = os.path.join(save_dir, f"timestep_attention_heatmap_block_{block_idx}.png")
        plt.savefig(img_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存时间步注意力图: {img_path} (形状: {timestep_attention.shape})")

        # 绘制二部图（修改为垂直布局）
        bipartite_path = os.path.join(save_dir, f"bipartite_block_{block_idx}.png")
        plot_attention_bipartite(
            timestep_attention,
            bipartite_path,
            block_idx
        )
        print(f"✅ 二部图已保存到: {bipartite_path}")

        return csv_path, img_path

    else:
        print(f"⚠️ 不支持的注意力维度: {attention_scores.dim()}")
        return None, None


def plot_spatial_attention(attention_scores, save_dir, block_idx):
    print(f"块 {block_idx} 原始空间注意力得分形状: {attention_scores.shape}")

    if attention_scores.dim() == 4:
        avg_attention = attention_scores.mean(dim=1)[0].detach().cpu().numpy()

        csv_path = os.path.join(save_dir, f"spatial_attention_block_{block_idx}.csv")
        pd.DataFrame(avg_attention).to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(avg_attention, annot=False, cmap="Reds", cbar=True, square=True)

        # 设置标题和轴标签字体大小
        # plt.title(f"空间注意力热力图 - 块 {block_idx}", fontsize=18)
        plt.xlabel("Key", fontsize=20)
        plt.ylabel("Query", fontsize=20)

        # 设置刻度字体大小
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        # 设置颜色条字体大小
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

        img_path = os.path.join(save_dir, f"spatial_attention_heatmap_block_{block_idx}.png")
        plt.tight_layout()
        plt.savefig(img_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存空间注意力图: {img_path} (形状: {avg_attention.shape})")

        return csv_path, img_path
    else:
        print(f"⚠️ 不支持的注意力维度: {attention_scores.dim()}")
        return None, None

def plot_attention_over_time(attention_scores, save_dir, block_idx):
    """绘制随时间变化的注意力得分（仅适用于时间分析）"""
    # 确定时间步数量
    num_timesteps = 21  # 根据您的数据

    # 计算实际时间步的起始位置
    class_tokens_count = 4  # 类别token数量
    timestep_start_idx = class_tokens_count  # 时间步从第4个位置开始

    # 提取第一个类别token对时间步的注意力
    if attention_scores.dim() == 4:  # [batch, heads, query_len, key_len]
        # 取第一个样本、第一个注意力头
        cls_attention = attention_scores[0, 0, 0, timestep_start_idx:timestep_start_idx + num_timesteps]
    elif attention_scores.dim() == 3:  # [heads, query_len, key_len]
        cls_attention = attention_scores[0, 0, timestep_start_idx:timestep_start_idx + num_timesteps]
    elif attention_scores.dim() == 2:  # [query_len, key_len]
        cls_attention = attention_scores[0, timestep_start_idx:timestep_start_idx + num_timesteps]
    else:
        print(f"⚠️ 不支持的注意力维度: {attention_scores.dim()}")
        return None

    cls_attention = cls_attention.detach().cpu().numpy()

    # 确保长度正确
    if len(cls_attention) != num_timesteps:
        print(f"⚠️ 时间步数量不匹配: 期望{num_timesteps}, 实际{len(cls_attention)}")
        return None

    # 创建时间序列
    time_steps = np.arange(num_timesteps)

    # 绘制折线图
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, cls_attention, marker='o', linestyle='-', color='b')
    # plt.title(f'CLS Token对时间步的注意力 - 块 {block_idx}', fontsize=16)
    plt.xlabel('observation time t')
    plt.ylabel('CLS attention scores')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, num_timesteps, step=1))

    # 保存图像
    img_path = os.path.join(save_dir, f"cls_attention_block_{block_idx}.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存CLS注意力图: {img_path}")

    return img_path


# ===========================================
# 主函数（修改版，添加门控权重保存和可视化）
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
    attention_dir = os.path.join(SAVE_DIR, f"{ANALYSIS_MODE}_attention_scores")
    os.makedirs(attention_dir, exist_ok=True)
    print(f"📁 注意力分析结果将保存在: {attention_dir}")

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

        # 转换为张量并调整维度顺序
        inputs = torch.tensor(model_input, dtype=torch.float32)
        inputs = inputs.unsqueeze(0)  # 添加批次维度 [1, T, C, H, W]
        inputs = inputs.to(device)
        print(f"📦 输入张量形状: {inputs.shape}")

        # 设置模型分析模式
        net.set_analysis_mode(ANALYSIS_MODE)

        # 运行模型前向传播 - 捕获注意力得分
        print(f"运行模型前向传播并捕获{ANALYSIS_MODE}注意力得分...")
        with torch.no_grad():
            logits, attention_scores_list = modified_forward_with_attention(inputs, net, mode=ANALYSIS_MODE)

        # 计算空间块数量
        num_spatial_blocks = (H // patch_size) * (W // patch_size)
        print(f"空间块数量: {num_spatial_blocks}")

        # 处理门控权重
        if ANALYSIS_MODE == 'spatial' and hasattr(net, 'gate_weights'):
            spatial_gate_weights = net.gate_weights
            print(f"✅ 收集到 {len(spatial_gate_weights)} 个空间门控权重块")

            # 保存CSV文件
            save_gate_weights(
                spatial_gate_weights,
                attention_dir,
                "spatial",
                num_positions=num_spatial_blocks
            )

            # 绘制柱状图
            plot_gate_weights(
                spatial_gate_weights,
                attention_dir,
                "spatial",
                num_positions=num_spatial_blocks,
                figsize=(12, 6)
            )
        elif ANALYSIS_MODE == 'temporal' and hasattr(net, 'gate_weights'):
            temporal_gate_weights = net.gate_weights
            print(f"✅ 收集到 {len(temporal_gate_weights)} 个时间门控权重块")

            # 保存CSV文件
            save_gate_weights(
                temporal_gate_weights,
                attention_dir,
                "temporal",
                num_positions=T  # 时间步数量
            )
        else:
            print("⚠️ 未检测到门控权重数据")

        # 计算每层空间块重要性
        if ANALYSIS_MODE == 'spatial' and attention_scores_list:
            print("计算每层空间块重要性...")

            # 为每一层计算块重要性
            block_importance_per_layer = compute_spatial_block_importance_per_layer(
                attention_scores_list,
                num_blocks=num_spatial_blocks
            )

            if block_importance_per_layer:
                # 创建DataFrame
                importance_df = pd.DataFrame(
                    block_importance_per_layer,
                    columns=[f"Block_{i}" for i in range(1, num_spatial_blocks + 1)]
                )

                # 添加层索引列
                importance_df.insert(0, 'Layer', range(1, len(block_importance_per_layer) + 1))

                # 保存为CSV
                importance_path = os.path.join(SAVE_DIR, "spatial_block_importance_per_layer.csv")
                importance_df.to_csv(importance_path, index=False)
                print(f"✅ 每层空间块重要性已保存至: {importance_path}")

                # 打印摘要信息
                print(f"共计算了 {len(block_importance_per_layer)} 层的空间块重要性")
                for layer_idx, imp_vec in enumerate(block_importance_per_layer):
                    print(f"第 {layer_idx + 1} 层重要性向量: {imp_vec}")
            else:
                print("⚠️ 未获取到空间块重要性数据")
        else:
            print("⚠️ 空间模式未检测到注意力得分")

        # 可视化注意力得分
        print(f"✅ 检测到 {len(attention_scores_list)} 个注意力块")
        for block_idx, attention_scores in enumerate(attention_scores_list):
            print(f"\n处理块 {block_idx + 1}/{len(attention_scores_list)}")

            if ANALYSIS_MODE == 'temporal':
                # 绘制时间注意力热力图和二部图
                plot_temporal_attention(attention_scores, attention_dir, block_idx)

                # 绘制随时间变化的注意力
                plot_attention_over_time(attention_scores, attention_dir, block_idx)
            else:
                # 绘制空间注意力热力图
                plot_spatial_attention(attention_scores, attention_dir, block_idx)
    else:
        print(f"⚠️ 未找到样本文件 {PICKLE_FILE}，请检查路径")

    print("\n✅ 所有注意力分析完成！结果保存在:", SAVE_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()