import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml
from fvcore.nn import FlopCountAnalysis
import time


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

# === 模型导入 ===
from models.UNet3D.unet3d import UNet3D
from models.UNet3D.unet3df import UNet3D_CSCL
from models.CropTypeMapping.models import FCN_CRNN
from models.BiConvRNN.biconv_rnn import BiRNNSequentialEncoder
from models.TSViT.TSViTdense import TSViT
from data.PASTIS24.data_transforms import Normalize
from data import get_dataloaders

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


# === 计算 FLOP ===
def calculate_flops(model, input_tensor):
    flops = FlopCountAnalysis(model, input_tensor)
    return flops.total()


# === 计算推断速度 ===
def evaluate_inference_speed(model, dataloader, device):
    model.eval()
    times = []
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating"):
            inputs = sample['inputs'].to(device)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            times.append(end_time - start_time)

    avg_inference_time = np.mean(times)
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
    return avg_inference_time


# === 计算内存消耗 ===
def evaluate_memory_usage(model, dataloader, device):
    model.eval()
    torch.cuda.empty_cache()  # 清空缓存
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs['inputs'].to(device)
            torch.cuda.reset_peak_memory_stats()  # 重置内存统计
            outputs = model(inputs)
            peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
            print(f"Peak memory usage during inference: {peak_memory:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Efficiency Analysis')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--device', default='0', type=str, help='GPU device ids (comma-separated)')
    parser.add_argument('--weights', required=True, help='Path to trained weights (e.g., best.pth)')
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(',')]
    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(args.config)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)

    # 加载模型
    net = get_model(config, device)
    load_from_checkpoint(net, args.weights, device)
    net.to(device)

    if len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids)

    # === 计算效率分析 ===
    # 模拟输入数据，确保符合模型输入形状
    dummy_input = torch.randn(1, 60, 11, 24, 24).to(device)  # 假设模型输入形状 (B=1, T=60, C=11, H=24, W=24)

    # 计算 FLOP
    flops = calculate_flops(net, dummy_input)
    print(f"Total FLOP: {flops}")

    # 计算推断速度
    avg_inference_time = evaluate_inference_speed(net, dataloaders['eval'], device)

    # 计算内存消耗
    evaluate_memory_usage(net, dataloaders['eval'], device)
#  python train_and_eval/predict_merge.py
#  --config configs/PASTIS24/TSViT_fold1.yaml --device 0
#  --weights  C:/Users/vipuser/Desktop/DeepSatModels-main/models/saved_models/PASTIS24_flod1/fold1_shijiankongjian/best.pth