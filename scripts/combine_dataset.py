import os
import shutil
import glob
import re
import argparse
from tqdm import tqdm

def combine_datasets(input_folders, output_folder):
    """
    将多个文件夹中的.rgba.png和.pkl文件合并到一个目标文件夹中
    
    Args:
        input_folders (list): 输入文件夹路径列表
        output_folder (str): 输出文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 用于跟踪目标文件夹中的文件数量
    file_count = 0
    
    # 遍历每个输入文件夹
    for folder in tqdm(input_folders, desc="处理文件夹"):
        # 获取当前文件夹中的所有.rgba.png文件
        png_files = glob.glob(os.path.join(folder, "*.rgba.png"))
        
        for png_file in tqdm(png_files, desc=f"处理{folder}中的文件", leave=False):
            # 获取对应的.pkl文件
            base_name = os.path.basename(png_file).replace(".rgba.png", "")
            pkl_file = os.path.join(folder, f"{base_name}.pkl")
            
            # 确保.pkl文件存在
            if not os.path.exists(pkl_file):
                print(f"警告：找不到与{png_file}对应的pkl文件，跳过此对")
                continue
            
            # 创建新的文件名
            new_base_name = f"{file_count:06d}"
            new_png_file = os.path.join(output_folder, f"{new_base_name}.rgba.png")
            new_pkl_file = os.path.join(output_folder, f"{new_base_name}.pkl")
            
            # 复制文件
            shutil.copy2(png_file, new_png_file)
            shutil.copy2(pkl_file, new_pkl_file)
            
            # 增加计数器
            file_count += 1
    
    print(f"合并完成！共合并了{file_count}对文件到{output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多个文件夹中的数据集")
    parser.add_argument("--input_folders", type=str, nargs='+', required=True, 
                        help="输入文件夹路径列表，例如'/path/to/output1 /path/to/output2'")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="输出文件夹路径")
    args = parser.parse_args()
    
    # 获取所有输入文件夹
    input_folders = args.input_folders
    
    if not input_folders:
        print("错误：未提供输入文件夹")
        exit(1)
    
    # 验证所有输入文件夹是否存在
    invalid_folders = [folder for folder in input_folders if not os.path.isdir(folder)]
    if invalid_folders:
        print(f"错误：以下文件夹不存在或不是有效的目录：")
        for folder in invalid_folders:
            print(f"  - {folder}")
        exit(1)
    
    print(f"将处理以下输入文件夹：")
    for folder in input_folders:
        print(f"  - {folder}")
    
    # 合并数据集
    combine_datasets(input_folders, args.output_folder)
