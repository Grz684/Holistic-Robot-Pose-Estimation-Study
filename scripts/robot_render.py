import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from lib.utils.urdf_robot import URDFRobot
from lib.utils.transforms import point_projection_from_3d_tensor
from lib.utils.vis import vis_3dkp_single_view
from lib.utils.geometries import rotmat_to_rot6d, rot6d_to_rotmat
from lib.dataset.const import INTRINSICS_DICT

def visualize_robot_on_image(image_path, robot_type, joint_angles, rotation, translation, 
                             output_folder, camera_code="azure", reference_keypoint_id=0):
    """
    渲染机器人模型到图像上并保存结果
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 初始化URDF机器人模型
    robot = URDFRobot(robot_type)
    
    # 加载图像
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    # 获取相机内参
    K = INTRINSICS_DICT[camera_code]
    K_mat = np.array([
                [K[0], 0, K[2]],
                [0, K[1], K[3]],
                [0, 0, 1]
            ])
    K_tensor = torch.tensor(K_mat).float()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 将数据移至设备
    joint_angles_tensor = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0).to(device)
    rotation_tensor = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0).to(device)
    translation_tensor = torch.tensor(translation, dtype=torch.float32).unsqueeze(0).to(device)
    K_tensor = K_tensor.to(device)
    
    image_resized = cv2.resize(image, (640, 480))
    image_tensor = torch.tensor(image_resized).float().unsqueeze(0).to(device)
    
    # 设置渲染器 - 直接使用相同的设置方式
    cpu_renderer = robot.set_robot_renderer(K_mat, scale=1, device="cpu")
    gpu_renderer = robot.set_robot_renderer(K_mat, scale=1, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取机器人网格
    robot_mesh_batch = robot.get_robot_mesh_list(joint_angles=joint_angles_tensor, renderer=cpu_renderer)
    
    # 使用原始代码中的渲染方法
    try:
        # 使用分割网络获取图像中的机器人分割掩码
        # seg_net = seg_mask_inference(K, camera_code)
        # seg_mask = seg_net(image_tensor * 255).detach()
        rendered_mask = robot.get_rendered_mask_single_image_at_specific_root(
            joint_angles_tensor[0], rotation_tensor[0], translation_tensor[0], 
            robot_mesh_batch[0], gpu_renderer, root=reference_keypoint_id
        )
        rendered_mask_np = rendered_mask.squeeze(0).cpu().numpy() * 255
        # rendered_mask_np = rendered_mask.cpu().numpy() * 255
        print(rendered_mask_np.shape)
        
    except Exception as e:
        print(f"Error rendering mask: {e}")
        # 创建空掩码
        rendered_mask_np = np.zeros((480, 640), dtype=np.uint8)
    
    # 保存渲染掩码
    cv2.imwrite(os.path.join(output_folder, 'rendered_mask.jpg'), rendered_mask_np)
    # cv2.imwrite(os.path.join(output_folder, 'segmentation_mask.jpg'), seg_mask_np)
    
    # 在原始调整大小后的图像上添加渲染
    image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, 'origin.jpg'), image_bgr)
    
    # 使用参考代码相同的方式创建叠加图像
    overlay = image_bgr.copy()
    # 将渲染掩码二值化
    mask_binary = (rendered_mask_np > 128).astype(np.uint8)
    # 创建红色覆盖层
    red_overlay = np.zeros_like(overlay)
    red_overlay[:, :, 2] = 255  # 红色通道
    
    # 应用透明度叠加
    alpha = 0.5
    mask_3channel = np.stack([mask_binary, mask_binary, mask_binary], axis=-1)
    overlay = cv2.addWeighted(overlay, 1.0, red_overlay * mask_3channel, alpha, 0)
    
    cv2.imwrite(os.path.join(output_folder, 'image_with_robot.jpg'), overlay)
    
    # 如果需要将渲染和分割掩码叠加 (类似参考代码中的stacks)
    stacks = np.zeros((480, 640, 3), dtype=np.uint8)
    stacks[:, :, 0] = rendered_mask_np  # 红色通道 - 渲染掩码
    # 蓝色通道留空，可以放置其他分割掩码
    # overlay[:, :, 2] = seg_mask_np       # 分割掩码在蓝色通道
    cv2.imwrite(os.path.join(output_folder, 'stack.jpg'), stacks)
    
    # 渲染3D关键点
    try:
        keypoints3d = robot.get_keypoints_root(joint_angles_tensor,
                                rotation_tensor,
                                translation_tensor,
                                root=reference_keypoint_id)
        
        # 确保K_tensor形状正确(添加批次维度)
        if len(K_tensor.shape) == 2:
            K_tensor = K_tensor.unsqueeze(0)
            
        keypoints2d = point_projection_from_3d_tensor(K_tensor, keypoints3d)

        print(f"Keypoints 3D: {keypoints3d}")
        print(f"Keypoints 2D: {keypoints2d}")
        
        # 在图像上绘制关键点
        keypoints_image = overlay.copy()
        for kp in keypoints2d[0].cpu().numpy():
            x, y = int(kp[0]/2), int(kp[1]/2)
            if 0 <= x < image_resized.shape[1] and 0 <= y < image_resized.shape[0]:
                cv2.circle(keypoints_image, (x, y), 5, (0, 255, 0), -1)
        
        cv2.imwrite(os.path.join(output_folder, 'image_with_keypoints.jpg'), keypoints_image)
        
        # 使用3D可视化函数
        # vis_3dkp_single_view(keypoints3d[0].cpu(), None, 
        #                     os.path.join(output_folder, 'keypoints3d_view1.jpg'), 
        #                     12, -20)
        # vis_3dkp_single_view(keypoints3d[0].cpu(), None, 
        #                     os.path.join(output_folder, 'keypoints3d_view2.jpg'), 
        #                     12, 0)
        # vis_3dkp_single_view(keypoints3d[0].cpu(), None, 
        #                     os.path.join(output_folder, 'keypoints3d_view3.jpg'), 
        #                     12, 20)
    except Exception as e:
        print(f"Error rendering keypoints: {e}")
    
    return {
        'rendered_mask': rendered_mask_np,
        'image_with_robot': overlay,
        'original_image': image_bgr
    }

# 重要的是保持一致性：如果编码（rotation_matrix_to_6d）使用的是前两列，
# 那么解码（从6D表示回到旋转矩阵）也应该相应地设计。只要整个系统保持一致，两种方法都可以工作，但它们不能混用。

# 提取 R^T 的前两行就相当于提取 R 的前两列，又由于旋转矩阵R 的转置等于 R 的逆，所以如果6d->mat和mat->6d方法不对应，
# 就会产生求逆的效果
def rotation_matrix_to_6d(rotation_matrix):
    """
    将3×3旋转矩阵转换为6D表示
    
    参数:
        rotation_matrix: 形状为 [3, 3] 的旋转矩阵
        
    返回:
        6D表示，形状为 [6]
    """
    # 提取旋转矩阵的前两行
    first_row = rotation_matrix[0, :]   # 第一行
    second_row = rotation_matrix[1, :]  # 第二行
    
    # 将两行连接成6D表示
    rot_6d = np.concatenate([first_row, second_row])
    
    return rot_6d

# 使用示例
if __name__ == "__main__":
    # 参数需要根据你的实际情况调整
    image_path = "data/try_dofbot_synth_train_dr/000307.rgba.png"
    data_path = "output_test/000096.pkl"
    robot_type = "dofbot"  # 使用你的机器人类型

    import pickle

    # # 加载 pickle 文件
    # with open(data_path, 'rb') as f:
    #     data = pickle.load(f)

    # all_joint_angles = data["dofbot_joint_names_positions"].values()
    # act_joint_angles = list(all_joint_angles)[:6]
    # c2b_rot = data["camera_ros_axes_to_robot_root_rot"]
    # b2c_rot = np.transpose(c2b_rot)
    # b2c_trans = data["keypoint_dict"]["/World/dofbot/link1"]["keypoint_positon"]
    # # print(data)
    
    # # 这些参数需要从你的模型中获取或自行指定
    # joint_angles = np.deg2rad(act_joint_angles)
    # rotation = rotation_matrix_to_6d(b2c_rot)  # 旋转矩阵，需要替换
    # translation = b2c_trans

    # print(f"joint_angles: {joint_angles}, rotation: {rotation}, translation: {translation}")
    joint_angles = np.array([-0.5017,  0.7287,  0.2723, -1.8464,  0.4003, -0.3971])
    rotation = np.array([ 0.8707,  0.4097,  0.4127, -0.1925, -0.5064,  0.9106])
    translation = np.array([0.0322, 0.0404, 0.9677])
    
    output_folder = "visualization_output"
    
    visualize_robot_on_image(
        image_path, 
        robot_type, 
        joint_angles, 
        rotation, 
        translation, 
        output_folder,
        camera_code="orbbec",
        reference_keypoint_id=1
    )