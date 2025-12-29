import os
import numpy as np
import cv2
from osgeo import gdal, gdalconst


def read_tif_with_gdal(tif_path):
    dataset = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    if dataset is None:
        raise IOError(f"无法打开: {tif_path}")

    bands = []
    for i in range(1, 4): 
        band = dataset.GetRasterBand(i)
        bands.append(band.ReadAsArray())

    img_array = np.stack(bands, axis=-1)
    return img_array, dataset


def save_tif_with_gdal(output_path, array, ref_dataset):
    driver = gdal.GetDriverByName('GTiff')

    out_ds = driver.Create(
        output_path,
        xsize=array.shape[1],
        ysize=array.shape[0],
        bands=3, 
        eType=gdalconst.GDT_Byte
    )

    out_ds.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_ds.SetProjection(ref_dataset.GetProjection())

    for i in range(3):
        band = out_ds.GetRasterBand(i + 1)
        band.WriteArray(array[:, :, i])
        band.FlushCache()

    del out_ds  


def image_augmentation(img):
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

    img_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(img_folder)
        for file in files if file.lower().endswith(".tif")
    ]

    for img_file in img_files:
        try:
            img, ref_dataset = read_tif_with_gdal(img_file)

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            augmented_images = image_augmentation(img)

            base_name = os.path.splitext(os.path.basename(img_file))[0]
            for idx, aug_img in enumerate(augmented_images):
                output_path = os.path.join(output_folder, f"{base_name}_{idx}.tif")
                save_tif_with_gdal(output_path, aug_img, ref_dataset)


        except Exception as e:
            print(f"❌ 处理失败: {os.path.basename(img_file)} | 错误: {str(e)}")


if __name__ == "__main__":

    main()
