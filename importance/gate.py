#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
门控权重可视化脚本 - 支持时间和空间模式
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

# --- 添加项目路径 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- 导入自定义模块 ---
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
SAVE_DIR = r"C:\Users\Think\Desktop\gate"
PICKLE_FILE = r"C:\Users\Think\Desktop\bq\bq_new_new\kuochong_30\64\total2\20369_1_0.pickle"
NUM_SAMPLES = 5
DEVICE_IDS = [0]
ANALYSIS_MODE = 'spatial'  # 可选 'temporal' 或 'spatial'


# ===========================================
# 修复后的数据处理函数
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
    关键修改：保持原始形状不变，并添加批次维度
    """
    doy_normalized = doys / 365.0
    doy_channel = doy_normalized[:, np.newaxis, np.newaxis, np.newaxis]
    doy_channel = np.broadcast_to(
        doy_channel,
        (doy_normalized.shape[0], 1, normalized_img.shape[2], normalized_img.shape[3])
    )

    model_input = np.concatenate([normalized_img, doy_channel], axis=1)
    return model_input.astype(np.float32)


# --- 时间特征处理 ---
def process_time_features(xt, device):
    """处理时间特征，避免索引错误"""
    # 确保时间特征在合理范围内
    xt = torch.clamp(xt * 365.0001, 0, 365)
    xt = xt.to(torch.int64)

    # 检查最大值是否超过365
    max_val = xt.max().item()
    if max_val >= 366:
        print(f"⚠️ 警告: 最大时间特征值 {max_val} 超过365，将被裁剪")
        xt = torch.clamp(xt, 0, 365)

    # 执行one-hot编码
    xt = F.one_hot(xt, num_classes=366).to(torch.float32)
    return xt


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

    # 准备patch嵌入 - 手动实现重排
    x = inputs[:, :, :-1]  # 移除时间特征通道，保留20个波段 [B, T, 20, H, W]

    # 确保空间维度能被patch_size整除
    assert H % net.patch_size == 0, f"高度 {H} 不能被 patch_size {net.patch_size} 整除"
    assert W % net.patch_size == 0, f"宽度 {W} 不能被 patch_size {net.patch_size} 整除"

    # 计算patch数量
    num_patches_h = H // net.patch_size
    num_patches_w = W // net.patch_size
    num_patches = num_patches_h * num_patches_w

    # 手动实现重排操作
    # 原始形状: [B, T, 20, H, W]
    # 目标形状: [B * num_patches, T, patch_size * patch_size * 20]
    # 使用unfold操作提取patch
    x = x.unfold(3, net.patch_size, net.patch_size)  # 在高度维度上展开
    x = x.unfold(4, net.patch_size, net.patch_size)  # 在宽度维度上展开

    # 现在形状为: [B, T, 20, num_patches_h, num_patches_w, patch_size, patch_size]
    # 调整维度顺序
    x = x.permute(0, 3, 4, 1, 2, 5, 6)  # [B, num_patches_h, num_patches_w, T, 20, patch_size, patch_size]

    # 合并patch和通道维度
    x = x.reshape(B * num_patches_h * num_patches_w, T, 20 * net.patch_size * net.patch_size)

    # 应用线性变换
    x = net.to_patch_embedding[1](x)  # [B*num_patches, T, dim]

    # 添加时间位置嵌入
    x = x.reshape(B, num_patches, T, net.dim)  # [B, num_patches, T, dim]
    x += temporal_pos_embedding.unsqueeze(1)  # [B, num_patches, T, dim]
    x = x.reshape(B * num_patches, T, net.dim)  # [B*num_patches, T, dim]

    # 添加时间token
    cls_temporal_tokens = net.temporal_token.repeat(B * num_patches, 1, 1)
    x = torch.cat((cls_temporal_tokens, x), dim=1)  # [B*num_patches, T+num_classes, dim]

    # ✅ 开启收集门控权重
    net.temporal_transformer.collect_gate_weights = True

    # 时间变换器 - 这里记录门控权重
    x = net.temporal_transformer(x)
    x = x[:, :net.num_classes]  # [B*num_patches, num_classes, dim]

    # 空间变换器
    x = x.reshape(B, num_patches, net.num_classes, net.dim)  # [B, num_patches, num_classes, dim]
    x = x.permute(0, 2, 1, 3)  # [B, num_classes, num_patches, dim]
    x = x.reshape(B * net.num_classes, num_patches, net.dim)  # [B*num_classes, num_patches, dim]

    # 确保空间位置嵌入大小匹配
    space_pos_embedding = net.space_pos_embedding[:, :num_patches] if net.space_pos_embedding.shape[
                                                                          1] > num_patches else net.space_pos_embedding
    x += space_pos_embedding

    # 应用dropout
    if hasattr(net, 'dropout'):
        x = net.dropout(x)

    if hasattr(net, 'collect_gate_weights'):
        net.collect_gate_weights = True
    if hasattr(net.space_transformer, 'collect_gate_weights'):
        net.space_transformer.collect_gate_weights = True

    # 空间变换器
    x = net.space_transformer(x)  # [B*num_classes, num_patches, dim]

    # MLP头部
    x = net.mlp_head(x.reshape(-1, net.dim))  # [B*num_classes*num_patches, patch_size**2]

    # 重塑输出
    x = x.reshape(B, net.num_classes, num_patches, net.patch_size ** 2)  # [B, num_classes, num_patches, patch_size**2]
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


# --- 可视化门控权重 ---
def plot_gate_weights(gate_weights, save_dir, block_idx, mode='spatial'):
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)
    mode_dir = os.path.join(save_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)

    attn_values = gate_weights['attn_weights']  # [S, D]
    mamba_values = gate_weights['mamba_weights']  # [S, D]

    # ✅ 保存CSV（每个patch一行，每列是通道）
    attn_df = pd.DataFrame(attn_values)
    mamba_df = pd.DataFrame(mamba_values)

    attn_csv = os.path.join(mode_dir, f'attn_gate_weights_block_{block_idx}.csv')
    mamba_csv = os.path.join(mode_dir, f'mamba_gate_weights_block_{block_idx}.csv')
    attn_df.to_csv(attn_csv, index=False)
    mamba_df.to_csv(mamba_csv, index=False)
    print(f"✅ CSV已保存: {attn_csv} / {mamba_csv}")

    # ✅ 热力图绘制
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
        print(f"✅ 热力图已保存: {save_path}")


# --- 处理权重数据 ---
def process_weights(weights_dict, mode='temporal'):
    attn_weights = weights_dict['attn']  # [B, L, D]
    mamba_weights = weights_dict['mamba']

    # 转换为NumPy
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.cpu().numpy()
    if isinstance(mamba_weights, torch.Tensor):
        mamba_weights = mamba_weights.cpu().numpy()

    # +++ 新增：时间模式下移除类别token +++
    if mode == 'temporal' and attn_weights.shape[1] > 4:
        # 去掉前4个类别token
        attn_weights = attn_weights[:, 4:, :]
        mamba_weights = mamba_weights[:, 4:, :]
        print(f"✅ 已移除前4个类别token，剩余时间步: {attn_weights.shape[1]}")

    # 根据模式处理权重
    if mode == 'temporal':
        # 时间模式: 在批次和通道维度取平均 [B, T, D] -> [T]
        attn_avg = np.mean(attn_weights, axis=(0, 2))
        mamba_avg = np.mean(mamba_weights, axis=(0, 2))
    else:
        # 空间模式: 在批次和通道维度取平均 [B, S, D] -> [S]
        # attn_avg = np.mean(attn_weights, axis=(0, 2))
        # mamba_avg = np.mean(mamba_weights, axis=(0, 2))
        attn_avg = np.mean(attn_weights, axis=0)  # → shape: [S, D]
        mamba_avg = np.mean(mamba_weights, axis=0)

    return {
        'attn_weights': attn_avg,
        'mamba_weights': mamba_avg
    }

# --- 主函数 ---
def main():
    print("=" * 50)
    print(f"模型配置文件: {CFG_PATH}")
    print(f"模型权重文件: {WEIGHTS_PATH}")
    print(f"结果保存目录: {SAVE_DIR}")
    print(f"分析模式: {ANALYSIS_MODE}")
    print(f"使用设备: {'GPU' if DEVICE_IDS else 'CPU'} {DEVICE_IDS}")
    print("=" * 50)

    # 0. 设备设置
    device = get_device(DEVICE_IDS, allow_cpu=True)

    # 1. 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    gate_weights_dir = os.path.join(SAVE_DIR, "gate_weights")
    os.makedirs(gate_weights_dir, exist_ok=True)
    print(f"📁 结果将保存在: {gate_weights_dir}")

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

    # 设置分析模式
    if hasattr(net, 'set_analysis_mode'):
        net.set_analysis_mode(ANALYSIS_MODE)
        print(f"✅ 设置分析模式: {ANALYSIS_MODE}")

    # 获取模型参数
    patch_size = getattr(net, 'patch_size', 16)
    print(f"ℹ️ 使用patch_size: {patch_size}")

    # 6. 加载样本数据
    if PICKLE_FILE and os.path.exists(PICKLE_FILE):
        # 从单个pickle文件加载样本
        print(f"🚀 加载样本: {PICKLE_FILE}")
        with open(PICKLE_FILE, 'rb') as f:
            data = pickle.load(f)
        img_data, labels, doys = data['img'], data['labels'], data['doy']

        # 打印原始形状信息
        print(
            f"📊 原始数据形状 - 时间步: {img_data.shape[0]}, 通道: {img_data.shape[1]}, 空间: {img_data.shape[2]}x{img_data.shape[3]}")

        # 应用自定义归一化
        normalized_img = custom_normalize(img_data, mean, std)

        # 准备模型输入（使用修改后的函数）
        model_input = prepare_model_input(normalized_img, doys)

        # 打印调整后的形状
        T, C, H, W = model_input.shape
        print(f"🔄 模型输入形状 - 时间步: {T}, 通道: {C}, 空间: {H}x{W}")
        print(f"ℹ️ 空间维度 {H}x{W} 应能被 {patch_size} 整除: {H % patch_size == 0 and W % patch_size == 0}")

        # 转换为张量并调整维度顺序以匹配模型期望
        # 模型期望维度顺序: [batch, time, channels, height, width]
        inputs = torch.tensor(model_input, dtype=torch.float32)  # [T, C, H, W]
        inputs = inputs.unsqueeze(0)  # 添加批次维度 [1, T, C, H, W]
        inputs = inputs.to(device)
        print(f"📦 输入张量形状: {inputs.shape}")

        # 运行模型前向传播 - 使用修复后的前向传播
        print("运行模型前向传播...")
        with torch.no_grad():
            logits = modified_forward(inputs, net)

        # 根据分析模式获取门控权重
        if ANALYSIS_MODE == 'temporal':
            # 时间模式
            if hasattr(net.temporal_transformer, 'gate_weights') and net.temporal_transformer.gate_weights:
                gate_weights = net.temporal_transformer.gate_weights
                print(f"✅ [时间] 检测到 {len(gate_weights)} 个门控权重块")

                for block_idx, weights_dict in enumerate(gate_weights):
                    # 处理权重数据
                    processed_weights = process_weights(weights_dict, mode='temporal')

                    # 可视化
                    plot_gate_weights(processed_weights, gate_weights_dir, block_idx, mode='temporal')
            else:
                print("⚠️ 警告: 时间转换器没有'gate_weights'属性或该属性为空")
        else:
            # 空间模式
            if hasattr(net.space_transformer, 'gate_weights') and net.space_transformer.gate_weights:
                gate_weights = net.space_transformer.gate_weights
                print(f"✅ [空间] 检测到 {len(gate_weights)} 个门控权重块")

                for block_idx, weights_dict in enumerate(gate_weights):
                    # 处理权重数据
                    processed_weights = process_weights(weights_dict, mode='spatial')

                    # 可视化
                    plot_gate_weights(processed_weights, gate_weights_dir, block_idx, mode='spatial')
            else:
                print("⚠️ 警告: 空间转换器没有'gate_weights'属性或该属性为空")
    else:
        # 使用验证集加载多个样本
        print("🚀 使用验证集加载样本...")
        if PICKLE_FILE:
            print(f"⚠️ 未找到样本文件 {PICKLE_FILE}，改用验证集")

        val_loader = dataloaders["eval"]
        sample_count = 0

        # 创建一个进度条
        progress = tqdm(total=NUM_SAMPLES, desc="处理样本")

        for inputs, labels in val_loader:
            if sample_count >= NUM_SAMPLES:
                break

            # 直接使用验证集原始维度顺序
            # 验证集输入形状: [batch, time, channels, height, width]
            inputs = inputs.to(device)
            print(f"📦 验证集输入张量形状: {inputs.shape}")

            # 运行模型前向传播
            with torch.no_grad():
                logits = net(inputs)

            # 根据分析模式获取门控权重
            if ANALYSIS_MODE == 'temporal':
                # 时间模式
                if hasattr(net.temporal_transformer, 'gate_weights') and net.temporal_transformer.gate_weights:
                    gate_weights = net.temporal_transformer.gate_weights
                    print(f"✅ [时间] 检测到 {len(gate_weights)} 个门控权重块")

                    # 处理每个样本
                    for sample_idx in range(inputs.size(0)):
                        if sample_count >= NUM_SAMPLES:
                            break

                        print(f"\n处理样本 {sample_count + 1}/{NUM_SAMPLES}")

                        for block_idx, weights_dict in enumerate(gate_weights):
                            # 处理权重数据
                            processed_weights = process_weights(weights_dict, mode='temporal')

                            # 可视化
                            plot_gate_weights(processed_weights, gate_weights_dir, block_idx, mode='temporal')

                        sample_count += 1
                        progress.update(1)
                else:
                    print("⚠️ 警告: 时间转换器没有'gate_weights'属性或该属性为空")
                    break
            else:
                # 空间模式
                if hasattr(net.space_transformer, 'gate_weights') and net.space_transformer.gate_weights:
                    gate_weights = net.space_transformer.gate_weights
                    print(f"✅ [空间] 检测到 {len(gate_weights)} 个门控权重块")

                    # 处理每个样本
                    for sample_idx in range(inputs.size(0)):
                        if sample_count >= NUM_SAMPLES:
                            break

                        print(f"\n处理样本 {sample_count + 1}/{NUM_SAMPLES}")

                        for block_idx, weights_dict in enumerate(gate_weights):
                            # 处理权重数据
                            processed_weights = process_weights(weights_dict, mode='spatial')

                            # 可视化
                            plot_gate_weights(processed_weights, gate_weights_dir, block_idx, mode='spatial')

                        sample_count += 1
                        progress.update(1)
                else:
                    print("⚠️ 警告: 空间转换器没有'gate_weights'属性或该属性为空")
                    break

        progress.close()

    print("\n✅ 所有门控权重可视化完成！结果保存在:", SAVE_DIR)
    print("=" * 50)


# --- 运行主函数 ---
if __name__ == "__main__":
    main()