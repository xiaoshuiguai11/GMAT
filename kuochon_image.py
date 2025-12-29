import os
import numpy as np
from osgeo import gdal
from scipy.ndimage import rotate
import numpy as np
import cv2
import numpy as np


input_root = r"C:\Users\Think\Desktop\bq\bq_new_new\raster"                 # 原始影像路径
output_root = r"C:\Users\Think\Desktop\bq\bq_new_new\kuochong_90\data\raster"      # 增强结果保存路径

def read_single_band_tif(path):
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"无法打开文件: {path}")
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    return array, dataset

def write_single_band_tif(array, ref_dataset, out_path):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(
        out_path,
        ref_dataset.RasterXSize,
        ref_dataset.RasterYSize,
        1,
        gdal.GDT_Float32  # ⭐ 使用 Float32 类型
    )
    out_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_dataset.SetProjection(ref_dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()
    out_dataset = None


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

def process_raster_folder(input_root, output_root):
    for dirpath, _, filenames in os.walk(input_root):
        for file in filenames:
            if file.endswith('.tif'):
                full_path = os.path.join(dirpath, file)
                try:
                    # 相对路径：用于提取 PATCH_TIME 和 TIME_BAND
                    rel_path = os.path.relpath(full_path, input_root)
                    parts = rel_path.split(os.sep)
                    if len(parts) != 2:
                        print(f"❌ 跳过无效路径结构: {rel_path}")
                        continue
                    patch_time = parts[0]  # PATCH_TIME
                    time_band = parts[1]   # TIME_BAND.tif

                    # 分离 PATCH 和 TIME
                    patch, time = patch_time.split("_", 1)

                    # 读取影像并增强
                    img, ref = read_single_band_tif(full_path)
                    aug_imgs = image_augmentation(img)

                    for idx, aug_img in enumerate(aug_imgs):
                        new_patch_time = f"{patch}_{idx}_{time}"

                        out_dir = os.path.join(output_root, new_patch_time)
                        os.makedirs(out_dir, exist_ok=True)

                        out_path = os.path.join(out_dir, time_band)
                        write_single_band_tif(aug_img, ref, out_path)
                        print(f"✅ 成功增强: {out_path}")

                except Exception as e:
                    print(f"❌ 处理失败: {full_path}，错误: {e}")

if __name__ == "__main__":
    process_raster_folder(input_root, output_root)

