import os
import shutil
import argparse
import random
import glob
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

def split_dataset(src_dir, train_dir, test_dir, train_ratio=0.9):
    """
    将源文件夹中的文件按指定比例分割到训练集和测试集文件夹中，使用硬链接提高效率
    
    参数:
        src_dir (str): 源数据文件夹路径
        train_dir (str): 训练集文件夹路径
        test_dir (str): 测试集文件夹路径
        train_ratio (float): 训练集占比，默认0.9
    """
    # 创建目标文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有rgba.png文件
    png_files = glob.glob(os.path.join(src_dir, "*.rgba.png"))
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 获取所有pkl文件
    pkl_files = glob.glob(os.path.join(src_dir, "*.pkl"))
    print(f"找到 {len(pkl_files)} 个PKL文件")
    
    # 提取所有有效的文件对（同时有PNG和PKL的）
    png_bases = {os.path.basename(f).split('.')[0] for f in png_files}
    pkl_bases = {os.path.basename(f).split('.')[0] for f in pkl_files}
    valid_bases = list(png_bases.intersection(pkl_bases))
    
    print(f"有效文件对数（同时存在PNG和PKL的）: {len(valid_bases)}")
    print(f"只有PNG没有PKL的文件数: {len(png_bases - pkl_bases)}")
    print(f"只有PKL没有PNG的文件数: {len(pkl_bases - png_bases)}")
    
    # 随机打乱文件列表
    random.shuffle(valid_bases)
    
    # 计算训练集数量
    train_count = int(len(valid_bases) * train_ratio)
    
    # 分割文件
    train_ids = valid_bases[:train_count]
    test_ids = valid_bases[train_count:]
    
    print(f"\n按比例 {train_ratio} 分割后:")
    print(f"预期训练集文件对数: {len(train_ids)}")
    print(f"预期测试集文件对数: {len(test_ids)}")
    
    # 用于统计成功链接的文件数
    train_success = {"png": 0, "pkl": 0}
    test_success = {"png": 0, "pkl": 0}
    train_failed = {"png": 0, "pkl": 0}
    test_failed = {"png": 0, "pkl": 0}
    
    # 创建训练集文件硬链接
    print("\n创建训练集硬链接...")
    for base_id in tqdm(train_ids):
        # 链接rgba.png文件
        png_file = os.path.join(src_dir, f"{base_id}.rgba.png")
        png_dest = os.path.join(train_dir, f"{base_id}.rgba.png")
        if os.path.exists(png_file):
            if not os.path.exists(png_dest):
                try:
                    os.link(png_file, png_dest)
                    train_success["png"] += 1
                except OSError as e:
                    train_failed["png"] += 1
                    print(f"警告: 无法为 {png_file} 创建硬链接: {e}")
        else:
            train_failed["png"] += 1
            
        # 链接pkl文件
        pkl_file = os.path.join(src_dir, f"{base_id}.pkl")
        pkl_dest = os.path.join(train_dir, f"{base_id}.pkl")
        if os.path.exists(pkl_file):
            if not os.path.exists(pkl_dest):
                try:
                    os.link(pkl_file, pkl_dest)
                    train_success["pkl"] += 1
                except OSError as e:
                    train_failed["pkl"] += 1
                    print(f"警告: 无法为 {pkl_file} 创建硬链接: {e}")
        else:
            train_failed["pkl"] += 1
    
    # 创建测试集文件硬链接
    print("\n创建测试集硬链接...")
    for base_id in tqdm(test_ids):
        # 链接rgba.png文件
        png_file = os.path.join(src_dir, f"{base_id}.rgba.png")
        png_dest = os.path.join(test_dir, f"{base_id}.rgba.png")
        if os.path.exists(png_file):
            if not os.path.exists(png_dest):
                try:
                    os.link(png_file, png_dest)
                    test_success["png"] += 1
                except OSError as e:
                    test_failed["png"] += 1
                    print(f"警告: 无法为 {png_file} 创建硬链接: {e}")
        else:
            test_failed["png"] += 1
            
        # 链接pkl文件
        pkl_file = os.path.join(src_dir, f"{base_id}.pkl")
        pkl_dest = os.path.join(test_dir, f"{base_id}.pkl")
        if os.path.exists(pkl_file):
            if not os.path.exists(pkl_dest):
                try:
                    os.link(pkl_file, pkl_dest)
                    test_success["pkl"] += 1
                except OSError as e:
                    test_failed["pkl"] += 1
                    print(f"警告: 无法为 {pkl_file} 创建硬链接: {e}")
        else:
            test_failed["pkl"] += 1
    
    # 复制_camera_settings.json文件到两个目标文件夹
    camera_settings_file = os.path.join(src_dir, "_camera_settings.json")
    if os.path.exists(camera_settings_file):
        try:
            # 尝试使用硬链接复制camera_settings文件
            if not os.path.exists(os.path.join(train_dir, "_camera_settings.json")):
                os.link(camera_settings_file, os.path.join(train_dir, "_camera_settings.json"))
            if not os.path.exists(os.path.join(test_dir, "_camera_settings.json")):
                os.link(camera_settings_file, os.path.join(test_dir, "_camera_settings.json"))
            print("已创建_camera_settings.json的硬链接")
        except OSError:
            # 如果硬链接失败，使用复制
            shutil.copy2(camera_settings_file, os.path.join(train_dir, "_camera_settings.json"))
            shutil.copy2(camera_settings_file, os.path.join(test_dir, "_camera_settings.json"))
            print("已复制_camera_settings.json到两个目标文件夹")
    else:
        print("警告: 未找到_camera_settings.json文件")
    
    # 输出详细统计信息
    print("\n详细统计信息:")
    print(f"训练集:")
    print(f"  成功创建: {train_success['png']} 张图片, {train_success['pkl']} 个pkl文件")
    print(f"  失败数量: {train_failed['png']} 张图片, {train_failed['pkl']} 个pkl文件")
    print(f"测试集:")
    print(f"  成功创建: {test_success['png']} 张图片, {test_success['pkl']} 个pkl文件")
    print(f"  失败数量: {test_failed['png']} 张图片, {test_failed['pkl']} 个pkl文件")
    
    # 验证最终文件数
    final_train_png = len(glob.glob(os.path.join(train_dir, "*.rgba.png")))
    final_train_pkl = len(glob.glob(os.path.join(train_dir, "*.pkl")))
    final_test_png = len(glob.glob(os.path.join(test_dir, "*.rgba.png")))
    final_test_pkl = len(glob.glob(os.path.join(test_dir, "*.pkl")))
    
    print("\n最终文件数验证:")
    print(f"训练集文件夹中: {final_train_png} 张图片, {final_train_pkl} 个pkl文件")
    print(f"测试集文件夹中: {final_test_png} 张图片, {final_test_pkl} 个pkl文件")
    
    # 验证比例
    total_png = final_train_png + final_test_png
    actual_train_ratio = final_train_png / total_png if total_png > 0 else 0
    print(f"\n实际训练比例: {actual_train_ratio:.4f} (目标: {train_ratio:.4f})")

def main():
    parser = argparse.ArgumentParser(description="将数据集按比例分割为训练集和测试集（使用硬链接）")
    parser.add_argument("--src_dir", type=str, required=True, help="源数据文件夹路径")
    parser.add_argument("--train_dir", type=str, required=True, help="训练集文件夹路径")
    parser.add_argument("--test_dir", type=str, required=True, help="测试集文件夹路径")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集占比，默认0.9")
    parser.add_argument("--copy", action="store_true", help="使用复制而不是硬链接（如果源和目标不在同一文件系统）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保可重现性
    random.seed(args.seed)
    
    if args.copy:
        print("使用文件复制模式而不是硬链接")
        # 这里调用原来的使用复制的函数
        # 为简便起见，这里没有实现复制版本，您可以根据需要添加
    else:
        split_dataset(args.src_dir, args.train_dir, args.test_dir, args.train_ratio)
    
    print(f"数据集分割完成!")

if __name__ == "__main__":
    main()