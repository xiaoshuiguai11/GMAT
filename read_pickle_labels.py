import os
import rasterio
import numpy as np
from collections import defaultdict, Counter
import shutil
from tqdm import tqdm


def get_class_counts(tif_path):
    """读取 tif 文件并统计每个有效像素值的数量（排除nodata值）"""
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        nodata = src.nodata
        data = data.astype(np.int32)

        if nodata is not None:
            valid_mask = data != nodata
            valid_data = data[valid_mask]
        else:
            valid_data = data.flatten()

        return Counter(valid_data)


def compute_split_targets(total_counts, ratios=(0.8, 0.1, 0.1)):
    """精确计算分割目标并确保总和正确"""
    targets = []
    for ratio in ratios:
        split = {}
        for cls, count in total_counts.items():
            split[cls] = int(count * ratio)
        targets.append(split)

    # 处理余数：将剩余像素分配给最后一个分组
    for cls, count in total_counts.items():
        allocated = sum(t[cls] for t in targets)
        remainder = count - allocated
        if remainder > 0:
            targets[-1][cls] += remainder

    return targets[0], targets[1], targets[2]


def assign_files_optimized(file_class_counts, total_counts, train_target, val_target, test_target):
    """优化后的文件分配算法"""
    assigned = {'train': [], 'val': [], 'test': []}
    current = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }

    # 按文件对各类别剩余需求的贡献度排序
    files_sorted = []
    for file, counts in file_class_counts.items():
        score = 0
        for cls, cnt in counts.items():
            remaining = {
                'train': max(0, train_target[cls] - current['train'][cls]),
                'val': max(0, val_target[cls] - current['val'][cls]),
                'test': max(0, test_target[cls] - current['test'][cls])
            }
            total_remaining = sum(remaining.values())
            if total_remaining > 0:
                score += sum(cnt * (remaining[s] / total_remaining) for s in ['train', 'val', 'test'])
        files_sorted.append((score, file))

    # 按评分降序处理文件
    for _, file in sorted(files_sorted, key=lambda x: -x[0]):
        counts = file_class_counts[file]
        best_split = None
        best_score = -float('inf')

        # 评估每个split的适应性
        for split in ['train', 'val', 'test']:
            temp_score = 0
            for cls, cnt in counts.items():
                remaining = {
                    'train': train_target[cls] - current['train'][cls],
                    'val': val_target[cls] - current['val'][cls],
                    'test': test_target[cls] - current['test'][cls]
                }

                # 修复条件判断语法
                condition_met = (
                        (split == 'train' and (current[split][cls] + cnt <= train_target[cls])) or
                        (split == 'val' and (current[split][cls] + cnt <= val_target[cls])) or
                        (split == 'test' and (current[split][cls] + cnt <= test_target[cls]))
                )

                if condition_met:
                    contribution = cnt * (remaining[split] / (sum(remaining.values()) + 1e-6))
                else:
                    over = (current[split][cls] + cnt) - (
                        train_target[cls] if split == 'train'
                        else val_target[cls] if split == 'val'
                        else test_target[cls]
                    )
                    contribution = -over * 10  # 超额惩罚

                temp_score += contribution

            if temp_score > best_score:
                best_score = temp_score
                best_split = split

        # 分配文件并更新计数器
        assigned[best_split].append(file)
        for cls, cnt in counts.items():
            current[best_split][cls] += cnt

    return assigned  # 修复的返回语句位置


def split_tif_dataset_by_pixel_distribution(input_folder, output_root):
    tif_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]

    print(f"统计 {len(tif_files)} 个文件的像素分布...")
    file_class_counts = {}
    for tif in tqdm(tif_files):
        file_class_counts[tif] = get_class_counts(tif)

    total_counts = sum(file_class_counts.values(), Counter())
    train_target, val_target, test_target = compute_split_targets(total_counts)

    print("\n目标分配:")
    for cls in sorted(total_counts):
        print(
            f"类别 {cls}: 总像素={total_counts[cls]} | 目标: train={train_target[cls]}, val={val_target[cls]}, test={test_target[cls]}")

    print("\n优化文件分配中...")
    assigned_files = assign_files_optimized(file_class_counts, total_counts, train_target, val_target, test_target)

    print("\n验证分配结果:")
    for split in ['train', 'val', 'test']:
        split_counter = Counter()
        for f in assigned_files[split]:
            split_counter += file_class_counts[f]

        print(f"\n{split} ({len(assigned_files[split])} files):")
        for cls in sorted(total_counts):
            actual = split_counter[cls]
            target = train_target[cls] if split == 'train' else val_target[cls] if split == 'val' else test_target[cls]
            ratio = actual / total_counts[cls] if total_counts[cls] > 0 else 0
            print(f"  类别 {cls}: {actual}/{target} ({ratio:.1%})")

    print("\n复制文件中...")
    copy_files(assigned_files, output_root)


def copy_files(assigned_files, output_root):
    for split, files in assigned_files.items():
        split_dir = os.path.join(output_root, split)
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            shutil.copy(f, os.path.join(split_dir, os.path.basename(f)))


# 使用示例
input_folder = r'D:\data'  # 替换为你的 tif 文件夹路径
output_root = r'D:\1'

split_tif_dataset_by_pixel_distribution(input_folder, output_root)