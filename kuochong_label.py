import os
import numpy as np
import cv2
from osgeo import gdal, gdalconst

def read_tif_with_gdal(tif_path):
    dataset = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    if dataset is None:
        raise IOError(f"无法打开: {tif_path}")
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, dataset  # 返回原始数据和 gdal 数据集对象（用于获取地理信息）

def save_tif_with_gdal(output_path, array, ref_dataset):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_path,
        xsize=array.shape[1],
        ysize=array.shape[0],
        bands=1,
        eType=gdalconst.GDT_Byte  # 使用 8-bit 保存，可兼容 4-bit 标签
    )
    out_ds.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_ds.SetProjection(ref_dataset.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(array)
    out_ds.FlushCache()
    del out_ds  # 关闭数据集以写入文件


import cv2
import numpy as np

from scipy.ndimage import rotate

# def image_augmentation(img, angles=[90, 180, 270]):
#     """
#     图像增强函数，支持任意角度旋转
#     参数：
#         img: 输入图像数组 (H, W) 或 (H, W, C)
#         angles: 需要增强的旋转角度列表（单位：度）
#     """
#     augmented = []
#
#     # 基本翻转
#     augmented.append(np.flip(img, axis=1))  # 水平翻转
#     augmented.append(np.flip(img, axis=0))  # 垂直翻转
#     augmented.append(np.flip(np.flip(img, axis=1), axis=0))  # 水平垂直翻转
#
#     # 多角度旋转（使用最近邻插值保持离散值）
#     for angle in angles:
#         # 保持图像尺寸不变（reshape=False）
#         # 使用最近邻插值（order=0）保持类别完整性
#         # 填充值设为-1（可根据实际情况修改）
#         rotated = rotate(img,
#                          angle,
#                          reshape=False,
#                          order=0,
#                          mode='constant',
#                          cval=-1)
#         augmented.append(rotated)
#
#     return augmented

import cv2

import cv2
import numpy as np


def image_augmentation(img):
    """基于几何等价关系的图像增强"""
    transforms = []
    angle_flip_map = set()  # 记录已处理的等效角度

    # 基础图像尺寸
    h, w = img.shape[:2]

    # 核心处理函数
    def process_rotation(base_angle):
        """处理指定角度及其等效变换"""
        # 生成旋转本体
        if base_angle not in angle_flip_map:
            angle_flip_map.add((base_angle, 0))
            transforms.append(rotate_image(img, base_angle))

        # 生成翻转变体
        for flip_type in [1, 2, 3]:
            # 计算等效角度
            equiv_angle = calculate_equivalent_angle(base_angle, flip_type)
            normalized_angle = equiv_angle % 360

            # 记录等效关系
            if (normalized_angle, 0) not in angle_flip_map:
                angle_flip_map.add((normalized_angle, 0))
                transforms.append(rotate_image(img, normalized_angle))

    # 遍历所有基础角度
    for angle in range(0, 360, 90):
        process_rotation(angle)

    return transforms


def rotate_image(img, angle):
    """执行标准化旋转操作"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 特殊角度处理
    if angle % 90 == 0:
        angle_mapping = {
            0: cv2.ROTATE_90_CLOCKWISE,
            90: cv2.ROTATE_90_CLOCKWISE,  # 90°旋转
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        if angle in angle_mapping:
            # 确保使用正确的旋转常量
            rotated = cv2.rotate(img, angle_mapping[angle])
            return rotated

    # 通用角度旋转
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def calculate_equivalent_angle(angle, flip_type):
    """计算几何等效角度"""
    if flip_type == 1:  # 水平翻转等效
        return (180 - angle) % 360
    elif flip_type == 2:  # 垂直翻转等效
        return (360 - angle) % 360
    elif flip_type == 3:  # 双翻转等效
        return (angle + 180) % 360


def main():
    img_folder = r"C:\Users\Think\Desktop\bq\bq_new_new\bq_raster"
    output_folder = r"C:\Users\Think\Desktop\bq\bq_new_new\kuochong_90\data\bq"
    os.makedirs(output_folder, exist_ok=True)

    img_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(img_folder)
        for file in files if file.lower().endswith(".tif")
    ]

    for img_file in img_files:
        try:
            img, ref_dataset = read_tif_with_gdal(img_file)

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)  # 确保兼容写入

            augmented_images = image_augmentation(img)

            base_name = os.path.splitext(os.path.basename(img_file))[0]
            for idx, aug_img in enumerate(augmented_images):
                output_path = os.path.join(output_folder, f"{base_name}_{idx}.tif")
                save_tif_with_gdal(output_path, aug_img, ref_dataset)
            print(f"✅ 处理完成: {img_file}")
        except Exception as e:
            print(f"❌ 处理失败: {img_file}，错误: {str(e)}")

if __name__ == "__main__":
    main()
