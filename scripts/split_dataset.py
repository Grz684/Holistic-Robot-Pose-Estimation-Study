import os
import shutil
import argparse
import random
import glob
from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def split_dataset(src_dir, train_dir, test_dir, train_ratio=0.8):
    """
    将源文件夹中的文件按指定比例分割到训练集和测试集文件夹中
    
    参数:
        src_dir (str): 源数据文件夹路径
        train_dir (str): 训练集文件夹路径
        test_dir (str): 测试集文件夹路径
        train_ratio (float): 训练集占比，默认0.8
    """
    # 创建目标文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有rgba.png文件的基础名称（不包含扩展名）
    png_files = glob.glob(os.path.join(src_dir, "*.rgba.png"))
    # 提取基础ID编号（如"000001"）
    base_ids = [os.path.basename(f).split('.')[0] for f in png_files]
    
    # 确保没有重复
    base_ids = list(set(base_ids))
    
    # 随机打乱文件列表
    random.shuffle(base_ids)
    
    # 计算训练集数量
    train_count = int(len(base_ids) * train_ratio)
    
    # 分割文件
    train_ids = base_ids[:train_count]
    test_ids = base_ids[train_count:]
    
    print(f"总文件对数: {len(base_ids)}")
    print(f"训练集文件对数: {len(train_ids)}")
    print(f"测试集文件对数: {len(test_ids)}")
    
    # 复制训练集文件
    for base_id in train_ids:
        # 复制rgba.png文件
        png_file = os.path.join(src_dir, f"{base_id}.rgba.png")
        if os.path.exists(png_file):
            shutil.copy2(png_file, os.path.join(train_dir, f"{base_id}.rgba.png"))
        else:
            print(f"警告: 未找到文件 {png_file}")
        
        # 复制pkl文件
        pkl_file = os.path.join(src_dir, f"{base_id}.pkl")
        if os.path.exists(pkl_file):
            shutil.copy2(pkl_file, os.path.join(train_dir, f"{base_id}.pkl"))
        else:
            print(f"警告: 未找到文件 {pkl_file}")
    
    # 复制测试集文件
    for base_id in test_ids:
        # 复制rgba.png文件
        png_file = os.path.join(src_dir, f"{base_id}.rgba.png")
        if os.path.exists(png_file):
            shutil.copy2(png_file, os.path.join(test_dir, f"{base_id}.rgba.png"))
        else:
            print(f"警告: 未找到文件 {png_file}")
        
        # 复制pkl文件
        pkl_file = os.path.join(src_dir, f"{base_id}.pkl")
        if os.path.exists(pkl_file):
            shutil.copy2(pkl_file, os.path.join(test_dir, f"{base_id}.pkl"))
        else:
            print(f"警告: 未找到文件 {pkl_file}")
    
    # 复制_camera_settings.json文件到两个目标文件夹
    camera_settings_file = os.path.join(src_dir, "_camera_settings.json")
    if os.path.exists(camera_settings_file):
        shutil.copy2(camera_settings_file, os.path.join(train_dir, "_camera_settings.json"))
        shutil.copy2(camera_settings_file, os.path.join(test_dir, "_camera_settings.json"))
        print("已复制_camera_settings.json到两个目标文件夹")
    else:
        print("警告: 未找到_camera_settings.json文件")
        
    # 统计实际复制的文件数量
    train_png_count = len(glob.glob(os.path.join(train_dir, "*.rgba.png")))
    train_pkl_count = len(glob.glob(os.path.join(train_dir, "*.pkl")))
    test_png_count = len(glob.glob(os.path.join(test_dir, "*.rgba.png")))
    test_pkl_count = len(glob.glob(os.path.join(test_dir, "*.pkl")))
    
    print(f"\n实际复制文件统计:")
    print(f"训练集: {train_png_count} 张图片, {train_pkl_count} 个pkl文件")
    print(f"测试集: {test_png_count} 张图片, {test_pkl_count} 个pkl文件")

def main():
    parser = argparse.ArgumentParser(description="将数据集按比例分割为训练集和测试集")
    parser.add_argument("--src_dir", type=str, required=True, help="源数据文件夹路径")
    parser.add_argument("--train_dir", type=str, required=True, help="训练集文件夹路径")
    parser.add_argument("--test_dir", type=str, required=True, help="测试集文件夹路径")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集占比，默认0.8")
    
    args = parser.parse_args()
    
    split_dataset(args.src_dir, args.train_dir, args.test_dir, args.train_ratio)
    print(f"数据集分割完成! 训练集比例: {args.train_ratio}, 测试集比例: {1-args.train_ratio}")

if __name__ == "__main__":
    main()
