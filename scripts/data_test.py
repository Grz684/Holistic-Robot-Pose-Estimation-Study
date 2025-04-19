import os
import pickle
import glob
import numpy as np
import math

def read_specific_pkl_file(pkl_file_path):
    """
    读取指定pkl文件的dofbot_joint_names_positions字段，并转换为弧度
    
    Args:
        pkl_file_path: 指定的pkl文件路径
        
    Returns:
        list: 关节角度数组(弧度)，如果读取失败则返回None
    """
    print(f"读取文件: {os.path.basename(pkl_file_path)}")
    
    try:
        # 打开并读取pkl文件
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 尝试获取dofbot_joint_names_positions字段
        if 'dofbot_joint_names_positions' in data:
            joint_data = data['dofbot_joint_names_positions']
            print(f"关节数据: {joint_data}")
            
            # 提取6个关节角度到数组（度数）
            joint_angles_deg = [
                joint_data.get('joint1', 0),
                joint_data.get('joint2', 0),
                joint_data.get('joint3', 0),
                joint_data.get('joint4', 0),
                joint_data.get('joint5', 0),
                joint_data.get('left_joint_1', 0)
            ]
            
            # 将角度转换为弧度
            joint_angles_rad = np.deg2rad(joint_angles_deg).tolist()
            
            print(f"提取的6个关节角度(度): {joint_angles_deg}")
            print(f"转换为弧度: {joint_angles_rad}")
            print("-" * 50)
            return joint_angles_rad
        else:
            print(f"文件 {os.path.basename(pkl_file_path)} 中不存在 'dofbot_joint_names_positions' 字段")
    
    except Exception as e:
        print(f"读取文件 {pkl_file_path} 时出错: {str(e)}")
    
    return None

def check_joint_angles_in_bounds(joint_angles, bounds):
    """
    检查关节角度是否在指定的范围内
    
    Args:
        joint_angles: 关节角度数组(弧度)
        bounds: 关节角度的上下限
        
    Returns:
        bool: 所有关节角度都在范围内返回True，否则返回False
        list: 超出范围的关节信息
    """
    all_in_bounds = True
    out_of_bounds_joints = []
    
    for i, (angle, bound) in enumerate(zip(joint_angles, bounds)):
        if angle < bound[0] or angle > bound[1]:
            all_in_bounds = False
            out_of_bounds_joints.append((i+1, angle, bound))
    
    return all_in_bounds, out_of_bounds_joints

def read_and_check_joint_data():
    """
    读取目录下所有pkl文件，并在读取的同时检查关节角度是否在范围内
    只输出超出限制的文件信息
    """
    # 指定目录路径
    data_dir = "data/dofbot_synth_train_dr"
    
    # 获取所有pkl文件
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"未在 {data_dir} 目录下找到任何pkl文件")
        return
    
    # 定义关节角度范围
    bounds = [[-1.5707999e+00,  1.5707999e+00],
              [-1.0995574e+00,  2.0420351e+00],
              [-2.3736477e+00,  7.6794487e-01],
              [-3.2288592e+00, -8.7266460e-02],
              [-1.5707999e+00,  1.5707999e+00],
              [-6.1086524e-01,  1.0471976e+00]]
    
    # 统计变量
    total_files = len(pkl_files)
    out_of_bounds_count = 0
    out_of_bounds_files = []
    all_joint_angles = []
    
    print(f"\n开始读取并检查 {total_files} 个pkl文件...\n")
    
    # 遍历所有pkl文件
    for pkl_file in pkl_files:
        try:
            # 打开并读取pkl文件
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # 尝试获取dofbot_joint_names_positions字段
            if 'dofbot_joint_names_positions' in data:
                joint_data = data['dofbot_joint_names_positions']
                
                # 提取6个关节角度到数组（度数）
                joint_angles_deg = [
                    joint_data.get('joint1', 0),
                    joint_data.get('joint2', 0),
                    joint_data.get('joint3', 0),
                    joint_data.get('joint4', 0),
                    joint_data.get('joint5', 0),
                    joint_data.get('left_joint_1', 0)
                ]
                
                # 将角度转换为弧度
                joint_angles_rad = np.deg2rad(joint_angles_deg).tolist()
                
                # 检查关节角度是否在范围内
                in_bounds, out_of_bounds_info = check_joint_angles_in_bounds(joint_angles_rad, bounds)
                
                # 将当前文件的关节角度(弧度)添加到总列表中
                all_joint_angles.append(joint_angles_rad)
                
                # 如果有关节角度超出范围，输出信息
                if not in_bounds:
                    out_of_bounds_count += 1
                    filename = os.path.basename(pkl_file)
                    out_of_bounds_files.append(filename)
                    
                    print(f"\n文件 {filename} 的关节角度超出限制:")
                    for joint_idx, angle, bound in out_of_bounds_info:
                        print(f"关节{joint_idx}: {angle:.6f} rad, 超出范围 [{bound[0]:.6f}, {bound[1]:.6f}]")
                    print(f"完整关节数组: {joint_angles_rad}")
            else:
                print(f"文件 {os.path.basename(pkl_file)} 中不存在 'dofbot_joint_names_positions' 字段")
        
        except Exception as e:
            print(f"读取文件 {pkl_file} 时出错: {str(e)}")
    
    print(f"\n总共读取了 {total_files} 个文件的关节角度")
    print(f"其中有 {out_of_bounds_count} 个文件的关节角度超出限制 ({out_of_bounds_count/total_files*100:.2f}%)")
    # if out_of_bounds_count > 0:
    #     print("超出限制的文件列表:")
    #     for i, filename in enumerate(out_of_bounds_files, 1):
    #         print(f"{i}. {filename}")
    
    return all_joint_angles

if __name__ == "__main__":
    # 定义关节角度范围
    bounds = [[-1.5707999e+00,  1.5707999e+00],
              [-1.0995574e+00,  2.0420351e+00],
              [-2.3736477e+00,  7.6794487e-01],
              [-3.2288592e+00, -8.7266460e-02],
              [-1.5707999e+00,  1.5707999e+00],
              [-6.1086524e-01,  1.0471976e+00]]
    
    # 指定要读取的pkl文件路径
    # 可以使用绝对路径或相对路径
    specific_file = "data/dofbot_synth_train_dr/005061.pkl"  # 替换为实际的文件路径
    
    # 检查文件是否存在
    if os.path.exists(specific_file):
        # 读取指定的pkl文件
        joint_angles = read_specific_pkl_file(specific_file)
        
        # 检查关节角度是否在范围内
        if joint_angles:
            in_bounds, out_of_bounds_info = check_joint_angles_in_bounds(joint_angles, bounds)
            if not in_bounds:
                print(f"\n文件 {os.path.basename(specific_file)} 的关节角度超出限制:")
                for joint_idx, angle, bound in out_of_bounds_info:
                    print(f"关节{joint_idx}: {angle:.6f} rad, 超出范围 [{bound[0]:.6f}, {bound[1]:.6f}]")
    else:
        print(f"指定的文件 {specific_file} 不存在")
    
    # # 读取并检查所有pkl文件
    # print("\n开始读取并检查所有文件：")
    # all_joint_angles = read_and_check_joint_data()
    
    # if all_joint_angles:
    #     print(f"角度数组形状: {np.array(all_joint_angles).shape}")
