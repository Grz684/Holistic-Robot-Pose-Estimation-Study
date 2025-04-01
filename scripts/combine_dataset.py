import os
import shutil
import argparse
from tqdm import tqdm

def combine_datasets(input_folders, output_folder, use_hardlink=True):
    """
    将多个文件夹中的文件合并到一个目标文件夹中，
    每个文件夹中的文件编号都是000000到009999，
    合并时重新编号确保连续
    
    Args:
        input_folders (list): 输入文件夹路径列表
        output_folder (str): 输出文件夹路径
        use_hardlink (bool): 是否使用硬链接（True）或复制（False）
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    file_count = 0
    link_type = "链接" if use_hardlink else "复制"
    
    for i, folder in enumerate(tqdm(input_folders, desc="处理文件夹")):
        offset = i * 10000  # 每个文件夹偏移10000
        
        # 处理所有的.rgba.png和.pkl文件对
        for j in tqdm(range(10000), desc=f"处理{folder}中的文件", leave=False):
            old_base = f"{j:06d}"
            new_base = f"{j + offset:06d}"
            
            # 源文件路径
            png_file = os.path.join(folder, f"{old_base}.rgba.png")
            pkl_file = os.path.join(folder, f"{old_base}.pkl")
            
            # 目标文件路径
            new_png_file = os.path.join(output_folder, f"{new_base}.rgba.png")
            new_pkl_file = os.path.join(output_folder, f"{new_base}.pkl")
            
            # 检查源文件是否存在
            if not (os.path.exists(png_file) and os.path.exists(pkl_file)):
                continue
                
            # 根据选择创建链接或复制文件
            try:
                if use_hardlink:
                    os.link(png_file, new_png_file)
                    os.link(pkl_file, new_pkl_file)
                else:
                    shutil.copy2(png_file, new_png_file)
                    shutil.copy2(pkl_file, new_pkl_file)
                file_count += 2  # 计数PNG和PKL两个文件
            except Exception as e:
                print(f"处理文件{png_file}失败: {e}")
    
    print(f"合并完成！成功{link_type}了{file_count}个文件到{output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多个文件夹中的数据集并重新编号")
    parser.add_argument("--input_folders", type=str, nargs='+', required=True, 
                        help="输入文件夹路径列表，例如'/path/to/output1 /path/to/output2'")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="输出文件夹路径")
    parser.add_argument("--copy", action="store_true",
                        help="使用复制而不是硬链接")
    args = parser.parse_args()
    
    # 验证所有输入文件夹是否存在
    invalid_folders = [folder for folder in args.input_folders if not os.path.isdir(folder)]
    if invalid_folders:
        print(f"错误：以下文件夹不存在或不是有效的目录：")
        for folder in invalid_folders:
            print(f"  - {folder}")
        exit(1)
    
    print(f"将处理以下输入文件夹：")
    for i, folder in enumerate(args.input_folders):
        print(f"  {i+1}. {folder} -> 编号 {i*10000:06d} 到 {(i+1)*10000-1:06d}")
    
    # 合并数据集
    combine_datasets(args.input_folders, args.output_folder, not args.copy)