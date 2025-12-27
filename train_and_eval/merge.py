import os
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
import argparse
import glob


def merge_predictions(band1_root, pred_dir, output_tif, win_size, stride):
    """
    将预测的小块图像合并为完整的研究区图像

    参数:
        band1_root: 原始波段目录（用于获取地理信息）
        pred_dir: 预测小块图像目录
        output_tif: 输出TIFF文件路径（必须是文件路径）
        win_size: 滑窗大小
        stride: 滑窗步长
    """
    # 确保输出路径是文件而不是目录
    if os.path.isdir(output_tif):
        raise ValueError(f"输出路径必须是文件路径，不能是目录: {output_tif}")

    # 获取原始影像尺寸和地理信息
    # 寻找第一个可用的波段文件
    sample_files = glob.glob(os.path.join(band1_root, "*.tif"))
    if not sample_files:
        raise FileNotFoundError(f"在 {band1_root} 中未找到任何波段文件")

    # 使用第一个找到的波段文件获取地理信息
    sample_path = sample_files[0]
    print(f"🌐 使用地理参考文件: {os.path.basename(sample_path)}")

    with rasterio.open(sample_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        profile = src.profile

    print(f"📐 原始影像尺寸: {height}×{width}")
    print(f"🧩 窗口大小: {win_size}, 步长: {stride}")

    # 创建空白的研究区数组
    full_pred = np.zeros((height, width), dtype=np.uint8)
    print(f"🖼️ 创建空白研究区数组: {full_pred.shape}")

    # 计算所有窗口位置
    positions = []
    for row_off in range(0, height, stride):
        for col_off in range(0, width, stride):
            if row_off + win_size <= height and col_off + win_size <= width:
                positions.append((row_off, col_off))

    print(f"🔢 总窗口数: {len(positions)}")

    # 获取预测文件列表并排序
    pred_files = sorted(
        glob.glob(os.path.join(pred_dir, "*.png")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1])
    )

    if len(pred_files) != len(positions):
        print(f"⚠️ 警告: 预测文件数({len(pred_files)})与窗口数({len(positions)})不匹配")

    # 遍历并填充预测结果
    for idx, (row_off, col_off) in enumerate(tqdm(positions, desc="合并预测图像")):
        if idx >= len(pred_files):
            print(f"⚠️ 跳过位置 ({row_off}, {col_off})：没有对应的预测文件")
            continue

        patch_path = pred_files[idx]

        try:
            # 读取预测小块
            patch_img = Image.open(patch_path)
            patch_arr = np.array(patch_img)

            # 将预测结果填充到研究区
            full_pred[row_off:row_off + win_size,
            col_off:col_off + win_size] = patch_arr
        except Exception as e:
            print(f"❌ 处理 {patch_path} 时出错: {str(e)}")

    # 更新TIFF配置文件
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        # nodata=3,
        transform=transform,
        crs=crs
    )

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    # 保存完整预测结果
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(full_pred, 1)

    print(f"✅ 合成完成! 结果保存至: {output_tif}")

    # 计算并显示类别统计
    unique, counts = np.unique(full_pred, return_counts=True)
    print("📊 预测类别统计:")
    for cls, cnt in zip(unique, counts):
        print(f"  类别 {cls}: {cnt} 像素 ({cnt / (height * width) * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并预测小块为完整研究区')
    parser.add_argument('--band1_root', required=True, help='原始波段目录')
    parser.add_argument('--pred_dir', required=True, help='预测小块图像目录')
    parser.add_argument('--output_tif', required=True, help='输出TIFF文件路径（必须包含文件名）')
    parser.add_argument('--win_size', type=int, default=64, help='滑窗大小')
    parser.add_argument('--stride', type=int, default=64, help='滑窗步长')

    args = parser.parse_args()

    merge_predictions(
        band1_root=args.band1_root,
        pred_dir=args.pred_dir,
        output_tif=args.output_tif,
        win_size=args.win_size,
        stride=args.stride
    )

#   python train_and_eval/merge.py  --band1_root  G:/REF_resample/  --pred_dir  D:/2/predict1/test_predictions/TSViT_fold5   --output_tif  D:/2/predict1/test_predictions/2.tif
#   --pred_dir D:/2/predict_unet3d/test_predictions/UNET3D --output_tif  D:/2/predict_unet3d/test_predictions/UNET3D/UNET3D.tif
#    --pred_dir C:/Users/Think/Desktop/DeepSatModels-main/models/saved_models/cahngji/UNET3Df/test_predictions/UNET3Df --output_tif  D:/2/predict_unet3df/test_predictions/UNET3Df/UNET3Df.tif
#    --pred_dir D:/2/predict_Unet2D/test_predictions/UNet2D_CLSTM  --output_tif  D:/2/predict_Unet2D/test_predictions/UNET2D.tif
#  --pred_dir  D:/2/tsvit_yuan/test_predictions/TSViT_fold5  --output_tif  D:/2/tsvit_yuan/test_predictions/tsvit_yuan.tif
#    --pred_dir  D:/2/predict_BiConvGRU/test_predictions/BiConvGRU   --output_tif  D:/2/predict_BiConvGRU/test_predictions/BiConvGRU.tif