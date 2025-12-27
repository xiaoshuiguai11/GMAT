import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import yaml
import time
import datetime
import json
from pathlib import Path

# === Model imports ===
from models.UNet3D.unet3d import UNet3D
from models.UNet3D.unet3df import UNet3D_CSCL
from models.CropTypeMapping.models import FCN_CRNN
from models.BiConvRNN.biconv_rnn import BiRNNSequentialEncoder
from models.TSViT.TSViTdense import TSViT
from data.PASTIS24.data_transforms import Normalize
from data import get_dataloaders
import torchprofile


def get_model(config, device):
    model_config = config['MODEL']
    if model_config['architecture'] == "UNET3Df":
        return UNet3D_CSCL(model_config).to(device)
    if model_config['architecture'] == "UNET3D":
        return UNet3D(model_config).to(device)
    if model_config['architecture'] == "UNET2D-CLSTM":
        return FCN_CRNN(model_config).cuda()
    if model_config['architecture'] == "ConvBiRNN":
        return BiRNNSequentialEncoder(model_config, device).to(device)
    if model_config['architecture'] == "TSViT":
        return TSViT(model_config).to(device)
    raise NameError(f"Model architecture '{model_config['architecture']}' not supported.")


def read_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件未找到: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"YAML 解析失败: {e}")
    return config


def get_device(device_ids, allow_cpu=True):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_ids[0]}")
    elif allow_cpu:
        return torch.device("cpu")
    else:
        raise EnvironmentError("没有可用 GPU 且未启用 CPU。")


def load_from_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型权重未找到: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✅ 成功加载权重: {checkpoint_path}")
    return checkpoint_path


def apply_color_palette(img):
    # 4 类颜色：黑、红、绿、蓝
    palette = [
                  0, 0, 0,  # class 0 - black
                  255, 0, 0,  # class 1 - red
                  0, 255, 0,  # class 2 - green
                  0, 0, 255  # class 3 - blue
              ] + [0] * (256 * 3 - 12)  # 填充剩余
    img.putpalette(palette)
    return img


def compute_flops(model, input_tensor1, input_tensor2, seq_lengths):
    """
    计算模型的FLOPs
    注意：由于模型需要多个输入，我们使用一个包装函数
    """

    def forward_wrapper(x1, x2, lengths):
        return model(x1, x2, lengths)

    # 使用torchprofile计算FLOPs
    flops = torchprofile.profile_macs(forward_wrapper, (input_tensor1, input_tensor2, seq_lengths))
    return flops


def save_model_stats_to_log(model_stats, log_dir, config_name):
    """
    保存模型统计信息到日志文件
    """
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"model_stats_{config_name}_{timestamp}.json"
    log_path = os.path.join(log_dir, log_filename)

    # 保存为JSON格式
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(model_stats, f, indent=2, ensure_ascii=False)

    print(f"📝 模型统计信息已保存至: {log_path}")
    return log_path


def save_inference_stats_to_log(inference_stats, log_dir, config_name):
    """
    保存推理统计信息到日志文件
    """
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"inference_stats_{config_name}_{timestamp}.json"
    log_path = os.path.join(log_dir, log_filename)

    # 保存为JSON格式
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(inference_stats, f, indent=2, ensure_ascii=False)

    print(f"📝 推理统计信息已保存至: {log_path}")
    return log_path


def test_and_save_predictions(net, dataloader, config, device, save_dir="test_predictions",
                              stats_log_dir=None, config_name=None):
    """
    测试并保存预测结果，同时记录推理性能
    """
    os.makedirs(save_dir, exist_ok=True)
    net.eval()

    # 初始化统计变量
    model_stats = {}
    inference_stats = {}

    # 添加推理时间测量
    total_inference_time = 0.0
    total_samples = 0
    batch_times = []

    with torch.no_grad():
        # 动态获取输入形状
        print("📏 获取输入形状...")
        first_batch = next(iter(dataloader))

        # 准备输入数据（保持原有的维度处理逻辑）
        inputs = first_batch['inputs'].to(device)
        inputs_backward = first_batch['inputs_backward'].to(device)
        seq_lengths = first_batch['seq_lengths'].to(device)

        # 训练时的维度顺序调整，保持与训练过程一致
        inputs_forward = inputs.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]
        inputs_backward = inputs_backward.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]

        input_shape_forward = inputs_forward.shape
        input_shape_backward = inputs_backward.shape

        model_stats['input_shape_forward'] = list(input_shape_forward)
        model_stats['input_shape_backward'] = list(input_shape_backward)

        # print(f"检测到前向输入形状: {input_shape_forward}")
        # print(f"检测到后向输入形状: {input_shape_backward}")

        # 使用实际输入形状进行预热
        print("🔥 GPU预热中...")
        dummy_forward = torch.randn_like(inputs_forward)
        dummy_backward = torch.randn_like(inputs_backward)
        dummy_lengths = torch.ones_like(seq_lengths) * seq_lengths.max()

        for _ in range(5):
            _ = net(dummy_forward, dummy_backward, dummy_lengths)

        # 计算 FLOPs
        print("📊 计算模型的 FLOPs...")
        flops = compute_flops(net, dummy_forward, dummy_backward, dummy_lengths)
        print(f"FLOPs: {flops:,}")

        # 保存FLOPs到统计信息
        model_stats['FLOPs'] = int(flops)
        model_stats['FLOPs_formatted'] = f"{flops:,}"
        model_stats['FLOPs_G'] = flops / 1e9

        # 重新开始迭代（包括第一个批次）
        dataloader_iter = iter(dataloader)

        for batch_idx in tqdm(range(len(dataloader)), desc="Running Inference"):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break

            inputs = batch['inputs'].to(device)
            inputs_backward = batch['inputs_backward'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            file_names = batch['file_name']

            # 训练时的维度顺序调整，保持与训练过程一致
            inputs_forward = inputs.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]
            inputs_backward = inputs_backward.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]

            # 测量推理时间
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            # 模型推理
            logits = net(inputs_forward, inputs_backward, seq_lengths)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()

            # 累计时间
            batch_time = end_time - start_time
            total_inference_time += batch_time
            batch_times.append(batch_time)
            total_samples += inputs.shape[0]

            # 调整维度顺序 (B, C, H, W) -> (B, H, W, C)
            logits = logits.permute(0, 2, 3, 1)

            # 获取预测结果
            pred = torch.argmax(logits, dim=-1).cpu().numpy()

            # 保存每个样本的预测结果
            for i in range(inputs.shape[0]):
                # 获取原始文件名
                original_name = os.path.splitext(file_names[i])[0]

                # 创建预测图像
                pred_i = pred[i]
                pred_img = Image.fromarray(pred_i.astype(np.uint8), mode='P')
                pred_img = apply_color_palette(pred_img)

                # 保存预测图像
                pred_img.save(os.path.join(save_dir, f"{original_name}.png"))

    # 计算推理性能统计
    avg_time_per_sample = total_inference_time / total_samples
    fps = total_samples / total_inference_time if total_inference_time > 0 else 0
    min_batch_time = min(batch_times) if batch_times else 0
    max_batch_time = max(batch_times) if batch_times else 0
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    std_batch_time = np.std(batch_times) if batch_times else 0

    # 保存推理统计信息
    inference_stats = {
        'total_inference_time_seconds': float(total_inference_time),
        'total_samples': int(total_samples),
        'average_time_per_sample_seconds': float(avg_time_per_sample),
        'inference_speed_fps': float(fps),
        'batch_time_statistics': {
            'min_batch_time_seconds': float(min_batch_time),
            'max_batch_time_seconds': float(max_batch_time),
            'average_batch_time_seconds': float(avg_batch_time),
            'std_batch_time_seconds': float(std_batch_time),
            'total_batches': len(batch_times)
        }
    }

    if torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        inference_stats['gpu_memory_usage_mb'] = float(max_memory_mb)
        print(f"GPU内存使用: {max_memory_mb:.2f} MB")

    # 打印统计信息
    print("=" * 50)
    print("📈 推理性能统计:")
    print(f"总推理时间: {total_inference_time:.2f}秒")
    print(f"总样本数: {total_samples}")
    print(f"平均每样本推理时间: {avg_time_per_sample:.4f}秒")
    print(f"推理速度: {fps:.2f}样本/秒")
    print(
        f"批次时间统计: 最小={min_batch_time:.4f}秒, 最大={max_batch_time:.4f}秒, 平均={avg_batch_time:.4f}秒, 标准差={std_batch_time:.4f}秒")
    print("=" * 50)

    print(f"✅ 预测结果已保存至: {save_dir}")

    # 保存推理统计到日志文件
    if stats_log_dir and config_name:
        save_inference_stats_to_log(inference_stats, stats_log_dir, config_name)

    return inference_stats, model_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TSViT Inference Only')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--device', default='0', type=str, help='GPU device ids (comma-separated)')
    parser.add_argument('--weights', required=True, help='Path to trained weights (e.g., best.pth)')
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(',')]
    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(args.config)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)

    # === Normalize statistics phase ===
    print("📊 开始统计训练集均值与标准差...")
    normalize_obj = None
    for t in dataloaders['train'].dataset.transform.transforms:
        if isinstance(t, Normalize):
            t.compute_stats = True
            normalize_obj = t
            break
    if normalize_obj is None:
        raise RuntimeError("未找到 Normalize 实例，请检查 transforms 顺序。")
    with torch.no_grad():
        for sample in tqdm(dataloaders['train'], desc="Accumulating mean/std"):
            _ = sample
    normalize_obj.compute_mean_std()
    normalize_obj.compute_stats = False
    print("✅ Normalize 统计完成")

    # === Model prediction ===
    net = get_model(config, device)

    # 计算模型参数量
    print("=" * 50)
    print(f"📊 模型架构: {config['MODEL']['architecture']}")

    # 如果是 DataParallel，需要特殊处理
    if len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids)
        total_params = sum(p.numel() for p in net.module.parameters())
        trainable_params = sum(p.numel() for p in net.module.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # 计算不可训练参数
    non_trainable_params = total_params - trainable_params

    # 保存模型统计信息
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    model_stats = {
        'model_architecture': config['MODEL']['architecture'],
        'total_parameters': int(total_params),
        'total_parameters_formatted': f"{total_params:,}",
        'total_parameters_M': total_params / 1e6,
        'trainable_parameters': int(trainable_params),
        'trainable_parameters_formatted': f"{trainable_params:,}",
        'non_trainable_parameters': int(non_trainable_params),
        'non_trainable_parameters_formatted': f"{non_trainable_params:,}",
        'config_file': args.config,
        'weights_file': args.weights,
        'device_ids': device_ids,
        'device': str(device),
        'timestamp': datetime.datetime.now().isoformat()
    }

    # 打印参数量信息
    print(f"📈 总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"⚙️  可训练参数量: {trainable_params:,}")
    print(f"📉 不可训练参数量: {non_trainable_params:,}")
    print("=" * 50)

    # 加载模型权重
    weights_path = load_from_checkpoint(net, args.weights, device)
    net.to(device)

    if len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids)

    # 创建保存路径
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    test_save_dir = os.path.join(
        config['CHECKPOINT']['save_path'],
        "test_predictions",
        config_name
    )

    # 创建日志目录
    stats_log_dir = os.path.join(
        config['CHECKPOINT']['save_path'],
        "stats_logs",
        config_name
    )

    # 运行测试并获取统计信息
    inference_stats, flops_stats = test_and_save_predictions(
        net,
        dataloaders['test'],
        config,
        device,
        save_dir=test_save_dir,
        stats_log_dir=stats_log_dir,
        config_name=config_name
    )

    # 合并所有统计信息
    all_stats = {
        **model_stats,
        **flops_stats,
        'inference_statistics': inference_stats
    }

    # 保存完整统计信息到日志文件
    save_model_stats_to_log(all_stats, stats_log_dir, config_name)

    print("✅ 推理完成，预测结果和统计信息已保存。")

#   python train_and_eval/predictt.py
#   --config configs/PASTIS24/BiConvGRU.yaml --device 0
#   --weights     C:/Users/vipuser/Desktop/DeepSatModels-main/models/saved_models/changji/BiconvGRU/best.pth