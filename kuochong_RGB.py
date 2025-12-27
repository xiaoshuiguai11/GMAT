import os
import numpy as np
import cv2
from osgeo import gdal, gdalconst


def read_tif_with_gdal(tif_path):
    """读取三通道TIFF文件"""
    dataset = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    if dataset is None:
        raise IOError(f"无法打开: {tif_path}")

    # 读取三个波段
    bands = []
    for i in range(1, 4):  # 假设是RGB三通道
        band = dataset.GetRasterBand(i)
        bands.append(band.ReadAsArray())

    # 转换为HWC格式 (高度, 宽度, 通道)
    img_array = np.stack(bands, axis=-1)
    return img_array, dataset


def save_tif_with_gdal(output_path, array, ref_dataset):
    """保存三通道TIFF文件"""
    driver = gdal.GetDriverByName('GTiff')

    # 创建三波段数据集
    out_ds = driver.Create(
        output_path,
        xsize=array.shape[1],
        ysize=array.shape[0],
        bands=3,  # 三波段
        eType=gdalconst.GDT_Byte
    )

    # 设置地理信息
    out_ds.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_ds.SetProjection(ref_dataset.GetProjection())

    # 分别写入每个波段
    for i in range(3):
        band = out_ds.GetRasterBand(i + 1)
        band.WriteArray(array[:, :, i])
        band.FlushCache()

    del out_ds  # 关闭数据集


def image_augmentation(img):
    """三通道图像增强"""
    augmented = []

    # 水平翻转
    augmented.append(cv2.flip(img, 1))
    # 垂直翻转
    augmented.append(cv2.flip(img, 0))
    # 水平+垂直翻转
    augmented.append(cv2.flip(img, -1))
    # 顺时针90度
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    # 180度
    augmented.append(cv2.rotate(img, cv2.ROTATE_180))
    # 逆时针90度
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    return augmented


def main():
    img_folder = r"C:\Users\Think\Desktop\bq\RGB\DATA_SUBSET"
    output_folder = r"C:\Users\Think\Desktop\bq\RGB\data_kuchong"
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有TIFF文件
    img_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(img_folder)
        for file in files if file.lower().endswith(".tif")
    ]

    # 处理每个文件
    for img_file in img_files:
        try:
            # 读取三通道数据
            img, ref_dataset = read_tif_with_gdal(img_file)

            # 确保数据类型为uint8
            if img.dtype != np.uint8:
                # 假设原始数据是0-255范围，否则需要归一化处理
                img = img.astype(np.uint8)

            # 数据增强
            augmented_images = image_augmentation(img)

            # 保存结果
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            for idx, aug_img in enumerate(augmented_images):
                output_path = os.path.join(output_folder, f"{base_name}_{idx}.tif")
                save_tif_with_gdal(output_path, aug_img, ref_dataset)

            print(f"✅ 成功处理: {os.path.basename(img_file)}")

        except Exception as e:
            print(f"❌ 处理失败: {os.path.basename(img_file)} | 错误: {str(e)}")


if __name__ == "__main__":
    main()