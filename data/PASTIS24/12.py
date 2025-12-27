import os
import argparse
import pickle
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from yaml import safe_load as load
from models.TSViT.TSViTdense import TSViT  # 导入实际模型定义


# ========== 1. 配置加载 ==========
def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        return load(f)


# ========== 2. 数据加载与处理 ==========
def load_pickle_tensor(pickle_path, config):
    with open(pickle_path, 'rb') as f:
        sample = pickle.load(f)

    # 获取配置参数
    img_res = config["MODEL"]["img_res"]
    num_channels = config["MODEL"]["num_channels"]

    # 原始数据维度 [T, C, H, W]
    img = sample['img']  # 时间序列图像
    doy = sample['doy']  # 时间特征

    # 数据填充到目标尺寸
    T, C, H, W = img.shape
    img_padded = np.zeros((T, C, img_res, img_res))
    img_padded[:, :, :H, :W] = img

    # 添加DOY通道 [T, 1, H, W]
    doy_exp = np.repeat(doy[:, None, None, None], img_res, axis=2)
    doy_exp = np.repeat(doy_exp, img_res, axis=3)

    # 合并通道 [T, C+1, H, W]
    x = np.concatenate([img_padded, doy_exp], axis=1)

    # 转换为模型输入格式 [B, T, C+1, H, W]
    x = torch.from_numpy(x).float().permute(1, 0, 2, 3).unsqueeze(0)
    label = sample.get("labels", None)

    return x, label


# ========== 3. 模型加载 ==========
def get_model(config, device):
    # 从配置构建模型参数
    model_params = {
        'img_res': config["MODEL"]["img_res"],
        'patch_size': config["MODEL"]["patch_size"],
        'num_channels': config["MODEL"]["num_channels"] + 1,  # 包含DOY通道
        'num_classes': config["MODEL"]["num_classes"],
        'dim': config["MODEL"]["dim"],
        'temporal_depth': config["MODEL"]["temporal_depth"],
        'spatial_depth': config["MODEL"]["spatial_depth"],
        'heads': config["MODEL"]["heads"],
        'dim_head': config["MODEL"]["dim_head"],
        'dropout': config["MODEL"]["dropout"]
    }

    model = TSViT(model_params)
    return model.to(device)


# ========== 4. 预测函数 ==========
def predict_single_pickle(pickle_path, config_path, checkpoint_path, output_dir, device_id=0):
    # 初始化配置
    config = read_yaml(config_path)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = get_model(config, device)
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 处理多GPU训练保存的权重
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # 加载并预处理数据
    inputs, label = load_pickle_tensor(pickle_path, config)
    inputs = inputs.to(device)

    # 执行预测
    with torch.no_grad():
        logits = model(inputs)  # [B, H, W, C]
        pred_mask = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pickle_path))[0]

    # 定义调色板（根据实际类别配置）
    palette = np.array([
        [0, 0, 0],  # 类别0：黑色
        [255, 0, 0],  # 类别1：红色
        [0, 255, 0],  # 类别2：绿色
        [0, 0, 255]  # 类别3：蓝色
    ], dtype=np.uint8).flatten()

    # 保存预测结果
    pred_path = os.path.join(output_dir, f"{base}_pred.png")
    img = Image.fromarray(pred_mask.astype(np.uint8), mode='P')
    img.putpalette(palette)
    img.save(pred_path)

    # 保存真实标签（如果存在）
    if label is not None:
        gt_path = os.path.join(output_dir, f"{base}_gt.png")
        img = Image.fromarray(label.astype(np.uint8), mode='P')
        img.putpalette(palette)
        img.save(gt_path)

    print(f"✅ 预测完成！结果保存至：{pred_path}")


# ========== 5. 入口函数 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSViT预测脚本")
    parser.add_argument("--pickle", required=True, help="输入.pickle文件路径")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--output_dir", default="predictions", help="输出目录")
    parser.add_argument("--device", default="0", help="GPU设备ID")

    args = parser.parse_args()

    # 路径验证
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"模型文件不存在：{args.checkpoint}")

    predict_single_pickle(
        pickle_path=args.pickle,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device_id=int(args.device)
    )


# # 使用示例
# python data/PASTIS24/12.py  --pickle C:/Users/Think/Desktop/bq/bq_new_new/kuochong_no/data/train/20369_0_0.pickle
#  --config C:/Users/Think/Desktop/DeepSatModels-main/configs/PASTIS24/TSViT_fold5.yaml
# --checkpoint C:/Users/Think/Desktop/DeepSatModels-main/models/saved_models/PASTIS24/TSViT_fold5/best.pth
#  --output_dir  C:/Users/Think/Desktop/DeepSatModels-main/models/saved_models/PASTIS24/predictions      --device 0
#
#
#     --device 0