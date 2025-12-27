import sys
import os

sys.path.insert(0, os.getcwd())
import argparse
import distutils.version

import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input

from sklearn.metrics import confusion_matrix

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
from tqdm import tqdm  # 新增导入语句
from data.PASTIS24.data_transforms import Normalize
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

import sys
import os
import pandas as pd
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
import distutils.version
from sklearn.metrics import confusion_matrix

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
from tqdm import tqdm  # 新增导入语句
from data.PASTIS24.data_transforms import Normalize
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

from fvcore.nn import FlopCountAnalysis
import time



# 在文件顶部添加导入
import sys
sys.path.append(os.getcwd())  # 确保可以导入utils模块
# from utils.gradient_analysis import SpectralGradientAnalyzer

# # 定义光谱波段名称（根据实际数据调整）
# BAND_NAMES = [
#     'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A',
#     'EVI', 'GCVI', 'GNDVI', 'NDVI', 'NDWI', 'NREDI1', 'NREDI2', 'NREDI3',
#     'OSAVI', 'RVI'
# ]
#
# # 定义类别名称（根据实际数据调整）
# CLASS_NAMES = [
#     "棉花", "小麦", "玉米", "背景"
# ]
#



def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):
    # ==================== 初始化日志系统 ====================
    log_dir = config['CHECKPOINT']['log_path']
    os.makedirs(log_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f'training_{timestamp}.log')

    # 配置日志格式和处理器
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    formatter = logging.Formatter(log_format)


    # 文件处理器（带滚动备份）
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10MB per file
        backupCount=5,  # 保留5个备份
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 获取根Logger并配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 禁用matplotlib等库的冗余日志
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # 记录初始化信息
    logger.info("=" * 60)
    logger.info(f"训练任务启动 | 设备: {device} | 时间: {timestamp}")
    logger.info("=" * 60)
    logger.info(f"日志文件路径: {os.path.abspath(log_filename)}")

    def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn):
        optimizer.zero_grad()
        outputs = net(sample['inputs'].to(device))
        outputs = outputs.permute(0, 2, 3, 1)
        ground_truth = loss_input_fn(sample, device)

        # 处理 ground_truth 可能为元组的情况
        if isinstance(ground_truth, tuple):
            if len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            elif len(ground_truth) == 3:
                labels, _, unk_masks = ground_truth
            else:
                labels = ground_truth[0]  # 默认取第一个元素
                unk_masks = None
        else:
            labels = ground_truth
            unk_masks = None

        # 将标签转换为 LongTensor
        labels = labels.long()

        # 计算损失，根据是否有掩码选择不同的损失函数调用方式
        if unk_masks is not None:
            loss = loss_fn['mean'](outputs, (labels, unk_masks))
        else:
            loss = loss_fn['mean'](outputs, labels)

        loss.backward()
        optimizer.step()

        # 返回处理后的标签和掩码，以便后续统计
        return outputs, labels, unk_masks, loss
        # ↑ 这里从返回 3 个值改为返回 4 个值

    def evaluate(net, evalloader, loss_fn, config, loss_input_fn):
        import numpy as np
        num_classes = config['MODEL']['num_classes']
        predicted_all = []
        labels_all = []
        losses_all = []
        input_gradients_list = []

        net.eval()
        device = next(net.parameters()).device

        for step, sample in enumerate(evalloader):
            inputs = sample['inputs'].to(device)
            inputs.requires_grad = True  # 开启输入梯度追踪

            logits = net(inputs)  # 输出形状可能为 [B, C, H, W] 或 [B, H, W, C]
            if logits.dim() == 4 and logits.shape[1] != num_classes:
                logits = logits.permute(0, 2, 3, 1)  # 转成 [B, H, W, C]
            elif logits.shape[1] == num_classes:
                logits = logits.permute(0, 2, 3, 1)

            _, predicted = torch.max(logits.data, -1)  # 预测类别 [B, H, W]

            ground_truth = loss_input_fn(sample, logits.device)
            if isinstance(ground_truth, tuple) and len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            elif isinstance(ground_truth, tuple) and len(ground_truth) == 3:
                labels, _, unk_masks = ground_truth
            else:
                labels = ground_truth
                unk_masks = None

            if unk_masks is not None:
                unk_masks = unk_masks.to(torch.bool)
                while unk_masks.dim() > labels.dim():
                    unk_masks = unk_masks.squeeze(1)
                mask_flat = unk_masks.view(-1)
                predicted_all.append(predicted.view(-1)[mask_flat].detach().cpu().numpy())
                labels_all.append(labels.view(-1)[mask_flat].detach().cpu().numpy())
                loss = loss_fn['all'](logits, (labels, unk_masks))
                if loss.shape == labels.shape:
                    losses_all.append(loss.view(-1)[mask_flat].detach().cpu().numpy())
                else:
                    losses_all.append(np.repeat(loss.mean().detach().cpu().numpy(), mask_flat.sum().item()))
            else:
                predicted_all.append(predicted.detach().cpu().numpy().reshape(-1))
                labels_all.append(labels.detach().cpu().numpy().reshape(-1))
                loss = loss_fn['all'](logits, labels)
                if loss.shape == labels.shape:
                    losses_all.append(loss.view(-1).detach().cpu().numpy())
                else:
                    losses_all.append(np.repeat(loss.mean().detach().cpu().numpy(), labels.numel()))

            # 计算类别预测分数对输入时间序列每个时间点和波段的梯度
            target_class = 0  # 这里示例用类别0，可根据需求调整
            net.zero_grad()
            if inputs.grad is not None:
                inputs.grad.zero_()
            if logits.dim() == 4:
                score = logits[0, :, :, target_class].sum()
            else:
                score = logits[0, target_class]
            score.backward()
            grads = inputs.grad[0].detach().cpu().numpy()  # 第0个样本梯度
            input_gradients_list.append(grads)

            # if step == 0:
            #     break

        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(target_classes, predicted_classes, labels=range(num_classes))
        print("Confusion Matrix:")
        print(cm)
        # 将混淆矩阵保存到日志
        logger.info("\n验证集混淆矩阵 (Confusion Matrix):")
        # 添加行列标签
        cm_df = pd.DataFrame(cm,
                             index=[f"实际类别 {i}" for i in range(num_classes)],
                             columns=[f"预测类别 {i}" for i in range(num_classes)])
        logger.info("\n" + cm_df.to_string())

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)
        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

        import os
        save_dir = config['CHECKPOINT']['save_path']
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "input_gradients_batch0.npy"), np.array(input_gradients_list))
        # === 新增：统计每个波段和时间点的重要性 ===
        grads_array = np.array(input_gradients_list)  # shape: [1, T, H, W, C]
        grads_mean = np.mean(np.abs(grads_array), axis=(0, 2, 3))  # shape: [T, C]

        # 每个波段在所有时间点的平均重要性
        band_importance = np.mean(grads_mean, axis=0)  # shape: [C]
        time_importance = np.mean(grads_mean, axis=1)  # shape: [T]

        # 保存为 CSV 文件
        band_names = [f"B{i + 1:02d}" for i in range(band_importance.shape[0])]  # 如 B01, B02,...
        band_df = pd.DataFrame({"Band": band_names, "Importance": band_importance})
        band_df.to_csv(os.path.join(save_dir, "band_importance.csv"), index=False)

        time_df = pd.DataFrame({"TimeIndex": np.arange(len(time_importance)), "Importance": time_importance})
        time_df.to_csv(os.path.join(save_dir, "time_importance.csv"), index=False)
        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": eval_metrics['macro'][0],
                           "Precision": eval_metrics['macro'][1],
                           "Recall": eval_metrics['macro'][2], "F1": eval_metrics['macro'][3],
                           "IOU": eval_metrics['macro'][4]},
                 "micro": {"Loss": losses.mean(), "Accuracy": eval_metrics['micro'][0],
                           "Precision": eval_metrics['micro'][1],
                           "Recall": eval_metrics['micro'][2], "F1": eval_metrics['micro'][3],
                           "IOU": eval_metrics['micro'][4]},
                 "class": {"Loss": class_loss, "Accuracy": eval_metrics['class'][0],
                           "Precision": eval_metrics['class'][1],
                           "Recall": eval_metrics['class'][2], "F1": eval_metrics['class'][3],
                           "IOU": eval_metrics['class'][4]}}

                )

    # 主训练参数
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
    logger.info(f"[DEBUG] L2正则化系数（从配置文件读取）: {weight_decay}")
    # 初始化模型
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)
    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)
    os.makedirs(save_path, exist_ok=True)
    copy_yaml(config)


    # 初始化优化器
    loss_input_fn = get_loss_data_input(config)
    loss_fn = {
        'all': get_loss(config, device, reduction=None),
        'mean': get_loss(config, device, reduction="mean")
    }
    optimizer = optim.AdamW(get_net_trainable_params(net), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(config, optimizer, len(dataloaders['train']))
    writer = SummaryWriter(save_path)
    BEST_IOU = 0

    for param_group in optimizer.param_groups:
        logger.info(f"[DEBUG] 优化器参数组: weight_decay = {param_group['weight_decay']}")




    # 主训练循环
    net.train()


    #
    # 步骤1: 进入统计量计算模式
    for transform in dataloaders['train'].dataset.transform.transforms:
        if isinstance(transform, Normalize):
            transform.compute_stats = True
            break

    # 步骤2: 遍历训练集累积统计量（需关闭数据增强）
    logger.info("开始计算数据集的均值和标准差...")
    with torch.no_grad():
        for sample in tqdm(dataloaders['train'], desc="累积统计量"):
            pass  # 数据加载时会自动调用Normalize的__call__方法

    # 步骤3: 计算最终均值和标准差
    transform.compute_mean_std()
    transform.compute_stats = False  # 切换回应用归一化模式
    logger.info(f"计算完成！均值: {transform.mean.squeeze()}, 标准差: {transform.std.squeeze()}")



    for epoch in range(1, num_epochs + 1):
        # ===== 训练阶段 =====
        epoch_loss = 0.0
        total_correct = 0
        total_pixels = 0
        TP = np.zeros(num_classes)
        FP = np.zeros(num_classes)
        FN = np.zeros(num_classes)

        for sample in dataloaders['train']:
            # 直接获取处理后的 labels 和 unk_masks
            outputs, labels, unk_masks, loss = train_step(net, sample, loss_fn, optimizer, device, loss_input_fn)
            # ↑ 这里从接收 3 个返回值改为接收 4 个返回值

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, -1)



            if unk_masks is not None:
                unk_masks = unk_masks.to(torch.bool)
                while unk_masks.dim() > labels.dim():
                    unk_masks = unk_masks.squeeze(1)
                assert unk_masks.shape == labels.shape, f"mask shape {unk_masks.shape}, label shape {labels.shape}"
                mask_flat = unk_masks.view(-1)
                total_correct += (predicted.view(-1)[mask_flat] == labels.view(-1)[mask_flat]).sum().item()
                total_pixels += mask_flat.sum().item()

                # 为TP/FP/FN统计也加mask，只在有效区域统计
                for c in range(num_classes):
                    class_mask = (labels == c) & unk_masks
                    TP[c] += ((predicted == c) & class_mask).sum().item()
                    FP[c] += ((predicted == c) & (~(labels == c)) & unk_masks).sum().item()
                    FN[c] += ((predicted != c) & class_mask).sum().item()
            else:
                total_correct += (predicted == labels).sum().item()
                total_pixels += labels.numel()
                for c in range(num_classes):
                    TP[c] += ((predicted == c) & (labels == c)).sum().item()
                    FP[c] += ((predicted == c) & (labels != c)).sum().item()
                    FN[c] += ((predicted != c) & (labels == c)).sum().item()

        # ===== 新增：计算训练mIoU =====
        iou = TP / (TP + FP + FN + 1e-8)  # 添加微小值防止除零
        train_miou = np.nanmean(iou)      # 忽略NaN值计算均值

        # ===== 输出训练指标 =====
        train_acc = total_correct / total_pixels
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        logger.info(f"\nEpoch {epoch}/{num_epochs} 训练指标:")
        logger.info(
            "Loss: %.4f | Acc: %.4f | mPrecision: %.4f | mRecall: %.4f | mF1: %.4f | mIoU: %.4f",
            epoch_loss / len(dataloaders['train']),
            train_acc,
            np.nanmean(precision),
            np.nanmean(recall),
            np.nanmean(f1),
            train_miou
        )
        # print(f"- Accuracy: {train_acc:.4f}")
        # print(f"- Macro-Precision: {np.nanmean(precision):.4f}")
        # print(f"- Macro-Recall: {np.nanmean(recall):.4f}")
        # print(f"- Macro-F1: {np.nanmean(f1):.4f}")

        # ===== 每两个epoch验证 =====
        if epoch % 2 == 0:
            un_labels, eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config, loss_input_fn)
            logger.info(f"\nEpoch {epoch}/{num_epochs} 验证指标:")
            # 宏平均指标
            logger.info(
                "macro"
                "- Loss: %.4f"
                "- Accuracy: %.4f"
                "- Precision: %.4f"
                "- Recall: %.4f"
                "- F1: %.4f"
                "- mIoU: %.4f",
                eval_metrics['macro']['Loss'],
                eval_metrics['macro']['Accuracy'],
                eval_metrics['macro']['Precision'],
                eval_metrics['macro']['Recall'],
                eval_metrics['macro']['F1'],
                eval_metrics['macro']['IOU']
            )

            # 微平均指标
            logger.info(
                "micro"
                "- Loss: %.4f"
                "- Accuracy: %.4f"
                "- Precision: %.4f"
                "- Recall: %.4f"
                "- F1: %.4f"
                "- mIoU: %.4f",
                eval_metrics['micro']['Loss'],
                eval_metrics['micro']['Accuracy'],
                eval_metrics['micro']['Precision'],
                eval_metrics['micro']['Recall'],
                eval_metrics['micro']['F1'],
                eval_metrics['micro']['IOU']
            )

            # 保存最佳模型
            if eval_metrics['macro']['IOU'] > BEST_IOU:
                model_state = net.module.state_dict() if len(local_device_ids) > 1 else net.state_dict()
                torch.save(model_state, f"{save_path}/best.pth")
                BEST_IOU = eval_metrics['macro']['IOU']
                logger.info(f"保存最佳模型，mIoU: {BEST_IOU:.4f}")

            net.train()  # 恢复训练模式




        # 更新学习率
        # scheduler.step_update(epoch * len(dataloaders['train']))

        # 更新学习率（仅当 scheduler 不为 None 时）
        if scheduler is not None:
            scheduler.step_update(epoch * len(dataloaders['train']))


from PIL import Image
import torchvision.transforms as transforms

def test_and_save_predictions(net, dataloader, config, device, save_dir="val_outputs"):  # 修改参数名
    os.makedirs(save_dir, exist_ok=True)
    net.eval()
    num_classes = config['MODEL']['num_classes']

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataloader, desc="Validating")):  # 修改描述
            inputs = sample['inputs'].to(device)
            logits = net(inputs).permute(0, 2, 3, 1)  # [B, H, W, C]
            pred = torch.argmax(logits, dim=-1).cpu().numpy()  # [B, H, W]

            labels = sample['labels'].cpu().numpy() if 'labels' in sample else None

            for i in range(inputs.size(0)):
                pred_i = pred[i]
                pred_img = Image.fromarray(pred_i.astype(np.uint8), mode='P')
                pred_img.save(os.path.join(save_dir, f"prediction_{idx}_{i}.png"))

                if labels is not None:
                    label_i = labels[i]
                    label_img = Image.fromarray(label_i.astype(np.uint8), mode='P')
                    label_img.save(os.path.join(save_dir, f"label_{idx}_{i}.png"))

    print(f"✅ 所有预测图像已保存至: {save_dir}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1', type=str,
                        help='gpu ids to use')
    parser.add_argument('--lin', action='store_true',
                        help='train linear classifier only')


    args = parser.parse_args()
    config_file = args.config
    print(args.device)
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device)

    # 加载模型权重
    load_from_checkpoint(net, f"{config['CHECKPOINT']['save_path']}/best.pth")
    net.to(device)




    # 使用 test dataloader 进行测试
    test_and_save_predictions(net, dataloaders['TEST'], config, device)




    dataloaders = get_dataloaders(config)

    net = get_model(config, device)


    # 1. 训练并保存最佳模型
    train_and_evaluate(net, dataloaders, config, device, lin_cls=lin_cls)

    # 2. 只加载最佳模型一次，并进行测试集泛化评估
    print("========== 开始测试集泛化评估 ==========")
    load_from_checkpoint(net, f"{config['CHECKPOINT']['save_path']}/best.pth")
    net.to(device)
    test_and_save_predictions(
        net,
        dataloaders['TEST'],
        config,
        device,
        save_dir=os.path.join(config['CHECKPOINT']['save_path'], "test_predictions")
    )
    print("✅ 测试集预测完成并已保存。")

 # python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/TSViT_fold1.yaml --device 0
 # python train_and_eval/segmentation_training_transf.py --config configs/MTLCC/UNet3D.yaml --device 0
 #  python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/UNet3D.yaml --device 0
 #  python train_and_eval/segmentation_training_transf.py --config configs/MTLCC/UNet3Df.yaml --device 0
 #   python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/UNet3Df.yaml --device 0
 # python train_and_eval/segmentation_training_transf.py --config configs/MTLCC/UNet2D_CLSTM.yaml --device 0
 #   python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/UNet2D_CLSTM.yaml --device 0
 # python train_and_eval/segmentation_training_transf.py --config configs/MTLCC/BiConvGRU.yaml --device 0
 #  python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/BiConvGRU.yaml --device 0
 # python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/TSViT_fold1.yaml --device 0
 # python train_and_eval/segmentation_training_transf.py --config configs/France/TSViT.yaml --device 0
 #
 #   python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/TSViT_fold5.yaml --device 0