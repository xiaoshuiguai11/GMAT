import os
import numpy as np
import cv2
from osgeo import gdal, gdalconst
import cv2
import numpy as np
from scipy.ndimage import rotate

def read_tif_with_gdal(tif_path):
    dataset = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    if dataset is None:
        raise IOError(f"无法打开: {tif_path}")
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, dataset  

def save_tif_with_gdal(output_path, array, ref_dataset):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_path,
        xsize=array.shape[1],
        ysize=array.shape[0],
        bands=1,
        eType=gdalconst.GDT_Byte 
    )
    out_ds.SetGeoTransform(ref_dataset.GetGeoTransform())
    out_ds.SetProjection(ref_dataset.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(array)
    out_ds.FlushCache()
    del out_ds  

def image_augmentation(img):
    transforms = []
    angle_flip_map = set() 

    h, w = img.shape[:2]

    def process_rotation(base_angle):

        if base_angle not in angle_flip_map:
            angle_flip_map.add((base_angle, 0))
            transforms.append(rotate_image(img, base_angle))

        for flip_type in [1, 2, 3]:
            equiv_angle = calculate_equivalent_angle(base_angle, flip_type)
            normalized_angle = equiv_angle % 360

            if (normalized_angle, 0) not in angle_flip_map:
                angle_flip_map.add((normalized_angle, 0))
                transforms.append(rotate_image(img, normalized_angle))

    for angle in range(0, 360, 90):
        process_rotation(angle)

    return transforms


def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    if angle % 90 == 0:
        angle_mapping = {
            0: cv2.ROTATE_90_CLOCKWISE,
            90: cv2.ROTATE_90_CLOCKWISE,  # 90°旋转
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        if angle in angle_mapping:
            rotated = cv2.rotate(img, angle_mapping[angle])
            return rotated

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def calculate_equivalent_angle(angle, flip_type):

    if flip_type == 1: 
        return (180 - angle) % 360
    elif flip_type == 2: 
        return (360 - angle) % 360
    elif flip_type == 3:  
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
                img = img.astype(np.uint8) 

            augmented_images = image_augmentation(img)

            base_name = os.path.splitext(os.path.basename(img_file))[0]
            for idx, aug_img in enumerate(augmented_images):
                output_path = os.path.join(output_folder, f"{base_name}_{idx}.tif")
                save_tif_with_gdal(output_path, aug_img, ref_dataset)
        except Exception as e:
            print(f"❌ 处理失败: {img_file}，错误: {str(e)}")

if __name__ == "__main__":
    main()

