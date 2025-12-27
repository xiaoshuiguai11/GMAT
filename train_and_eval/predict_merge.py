import os
import sys
import time
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
import torchprofile  # 用于计算FLOPs


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


def compute_flops(model, input_tensor):
    # 使用 torchprofile 来计算 FLOPs
    flops = torchprofile.profile_macs(model, input_tensor)
    return flops


def test_and_save_predictions(net, dataloader, config, device, save_dir="test_predictions"):
    os.makedirs(save_dir, exist_ok=True)
    net.eval()

    # 添加推理时间测量
    total_inference_time = 0.0
    total_samples = 0

    with torch.no_grad():
        # 动态获取输入形状
        print("📏 获取输入形状...")
        first_batch = next(iter(dataloader))
        inputs = first_batch['inputs'].to(device)
        input_shape = inputs.shape

        print(f"检测到输入形状: {input_shape}")

        # 使用实际输入形状进行预热
        print("🔥 GPU预热中...")
        dummy_input = torch.randn_like(inputs)  # 使用相同的形状
        for _ in range(5):
            _ = net(dummy_input)

        # 计算 FLOPs
        print("📊 计算模型的 FLOPs...")
        flops = compute_flops(net, dummy_input)
        print(f"FLOPs: {flops:,}")

        # 重新开始迭代（包括第一个批次）
        dataloader_iter = iter(dataloader)

        for batch_idx in tqdm(range(len(dataloader)), desc="Running Inference"):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break

            inputs = batch['inputs'].to(device)
            file_names = batch['file_name']

            # 测量推理时间
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            # 模型推理
            logits = net(inputs)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            # 累计时间
            batch_time = end_time - start_time
            total_inference_time += batch_time
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

    # 计算并打印推理速度统计
    # ... 后续代码保持不变 ...

    print(f"✅ 预测结果已保存至: {save_dir}")


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

    # === 动态 Normalize 统计阶段 ===
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


    # === 模型预测 ===
    net = get_model(config, device)

    # 计算并打印模型参数量
    print("=" * 50)
    print(f"📊 模型架构: {config['MODEL']['architecture']}")

    # 如果是 DataParallel，需要特殊处理
    if len(device_ids) > 1:
        # 多GPU模式下，先包装成DataParallel再计算
        net = nn.DataParallel(net, device_ids=device_ids)
        total_params = sum(p.numel() for p in net.module.parameters())
        trainable_params = sum(p.numel() for p in net.module.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"📈 总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"⚙️  可训练参数量: {trainable_params:,}")
    print(f"📉 不可训练参数量: {total_params - trainable_params:,}")
    print("=" * 50)

    load_from_checkpoint(net, args.weights, device)
    net.to(device)

    if len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids)

    # 创建更有序的保存路径
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    test_save_dir = os.path.join(
        config['CHECKPOINT']['save_path'],
        "test_predictions",
        config_name
    )

    test_and_save_predictions(
        net,
        dataloaders['test'],
        config,
        device,
        save_dir=test_save_dir
    )

    print("✅ 推理完成，预测结果已保存。")


#   python train_and_eval/predict_merge.py
#   --config configs/PASTIS24/TSViT_fold5.yaml --device 0
#   --weights  C:/Users/vipuser/Desktop/DeepSatModels-main/models/saved_models/PASTIS24/changji_kongjianshijian_TSViT_fold5/best.pth
#    --config configs/PASTIS24/UNET3D.yaml --device 0   --weights  C:/Users/Think/Desktop/DeepSatModels-main/models/saved_models/PASTIS24/changji_UNET3D/best.pth
#    --config configs/PASTIS24/UNET3Df.yaml   --device 0   --weights  C:/Users/Think/Desktop/DeepSatModels-main/models/saved_models/cahngji/UNET3Df/best.pth
#     --config configs/PASTIS24/UNet2D_CLSTM.yaml   --device 0   --weights C:/Users/Think/Desktop/DeepSatModels-main/models/saved_models/changji/changji_UNet2D_CLSTM/best.pth
#  --weights  C:/Users/Think/Desktop/DeepSatModels-main/models/saved_models/PASTIS24/changji_kongjianshijian_TSViT_fold5/best.pth
#    --config configs/PASTIS24/BiConvGRU.yaml --device 0     --weights    C:/Users/Think/Desktop/DeepSatModels-main/models/saved_models/changji/BiconvGRU/best.pth

#   --config configs/PASTIS24/TSViT_fold5.yaml --device 0    --weights    C:/Users/Think/Desktop/DeepSatModels-main/models/saved_models/changji/tsvit/best.pth

# C:/Users/vipuser/Desktop/DeepSatModels-main/models/saved_models/changji/unet3d/best.pth
# C:/Users/vipuser/Desktop/DeepSatModels-main/models/saved_models/PASTIS24_flod1/fold1_shijiankongjian/best.pth
#  C:/Users/vipuser/Desktop/DeepSatModels-main/models/saved_models/cahngji/UNET3Df/best.pth
# C:/Users/vipuser/Desktop/DeepSatModels-main/models/saved_models/changji/tsvit/best.pth
#  C:/Users/vipuser/Desktop/DeepSatModels-main/models/saved_models/changji/changji_UNet2D_CLSTM/best.pth
# C:/Users/vipuser/Desktop/DeepSatModels-main/models/saved_models/changji/BiconvGRU/best.pth

