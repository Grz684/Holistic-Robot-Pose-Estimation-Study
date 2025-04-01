import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 只让程序看到第4个GPU

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

print(f"Python search paths:")
for path in sys.path:
    print(f"  - {path}")

print(f"\nProject root: {project_root}")
print(f"Current working directory: {os.getcwd()}")
import json
from pathlib import Path
from PIL import Image
import PIL
import numpy as np
import torch
from lib.dataset.roboutils import bbox_transform, get_bbox, make_masks_from_det
import torch.nn.functional as F
from lib.utils.geometries import get_K_crop_resize
from lib.utils.urdf_robot import URDFRobot
from lib.models.full_net import get_rootNetwithRegInt_model
from scripts.test import make_cfg, cast
from lib.dataset.const import INITIAL_JOINT_ANGLE
from robot_render import rotation_matrix_to_6d
from lib.utils.transforms import point_projection_from_3d_tensor

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--exp_path', '-e', type=str, required=True)
parser.add_argument('--dataset', '-d', type=str, required=True, help= "e.g. panda_synth_test_dr") 
parser.add_argument('--known_joint', '-k', type=bool, default=False, help= "whether use gt joint for testing")
parser.add_argument('--model_name', '-m', type=str, default="curr_best_auc(add)_model", help= "model name") 
args = parser.parse_args()
args = make_cfg(args)

device_id = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_path = 'data/dofbot_synth_test_dr/000112.rgba.png'
base_dir = 'data/dofbot_synth_test_dr'

rgb_path = Path(rgb_path)
assert rgb_path
# 加载RGBA图像
image = Image.open(rgb_path)
# 检查图像是否有alpha通道
if image.mode == 'RGBA':
    # 方法1：丢弃alpha通道，只保留RGB
    rgb = np.asarray(image.convert('RGB'))
else:
    # 如果已经是RGB格式，直接使用
    rgb = np.asarray(image)

# permute(2, 0, 1) 会将其维度顺序调整为 (channels, height, width)（即 CHW 格式）。这是 PyTorch 处理图像数据的标准格式
images_original = torch.FloatTensor(rgb.copy()).permute(2,0,1)

data_path = rgb_path.with_suffix('').with_suffix('.pkl')
# annotations = json.loads(rgb_path.with_suffix('').with_suffix('.json').read_text())
import pickle
from collections import OrderedDict
# 加载 pickle 文件
with open(data_path, 'rb') as f:
    annotations = pickle.load(f)

base_dir = Path(base_dir)
camera_infos_path = base_dir / '_camera_settings.json'
h, w = rgb.shape[0], rgb.shape[1]
if camera_infos_path.exists():
    cam_infos = json.loads(camera_infos_path.read_text())
    assert len(cam_infos['camera_settings']) == 1
    # 定义相机内参
    cam_infos = cam_infos['camera_settings'][0]['intrinsic_settings']
    fx, fy, cx, cy = [cam_infos[k] for k in ('fx', 'fy', 'cx', 'cy')]
else:
    fx, fy = 320, 320
    cx, cy = w/2, h/2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
K_original = K.copy()   

keypoints_data = annotations["keypoint_dict"]
from lib.dataset.const import LINK_NAMES
link_names = LINK_NAMES['dofbot']
keypoints_data = annotations["keypoint_dict"]
filtered_keypoints_2d = []

urdf_robot_name = args.urdf_robot_name
robot = URDFRobot(urdf_robot_name)

for kp_name, kp_dict in keypoints_data.items():
    # Extract the link name from format like "/World/dofbot/link2"
    link_name = kp_name.split('/')[-1]
    
    if link_name in link_names:
        # This keypoint belongs to a link we're interested in
        filtered_keypoints_2d.append(np.array(kp_dict['keypoint_projection'])[None])

# Concatenate all the filtered keypoints
if filtered_keypoints_2d:
    keypoints_2d = np.concatenate(filtered_keypoints_2d, axis=0)
else:
    keypoints_2d = np.empty((0, *np.array(next(iter(keypoints_data.values()))['keypoint_projection']).shape))
# keypoints_2d = np.unique(keypoints_2d, axis=0)
print(f"keypoints_2d: {keypoints_2d}")

# -------------------------------------------
# filtered_keypoints_3d = []

# for kp_name, kp_dict in keypoints_data.items():
#     # 从格式如 "/World/dofbot/link2" 中提取链接名称
#     link_name = kp_name.split('/')[-1]
    
#     if link_name in link_names:
#         # 这个关键点属于我们感兴趣的链接
#         filtered_keypoints_3d.append(np.array(kp_dict['keypoint_positon'])[None])

# # 连接所有过滤后的关键点
# if filtered_keypoints_3d:
#     TCO_keypoints_3d = np.concatenate(filtered_keypoints_3d, axis=0)
# else:
#     TCO_keypoints_3d = np.empty((0, *np.array(next(iter(keypoints_data.values()))['keypoint_positon']).shape))

# print(f"TCO_keypoints_3d: {TCO_keypoints_3d}")

# all_joint_angles = annotations["dofbot_joint_names_positions"].values()
# act_joint_angles = list(all_joint_angles)[:6]
# c2b_rot = annotations["camera_ros_axes_to_robot_root_rot"]
# b2c_rot = np.transpose(c2b_rot)
# b2c_trans = annotations["keypoint_dict"]["/World/dofbot/link1"]["keypoint_positon"]
# # print(data)

# # 这些参数需要从你的模型中获取或自行指定
# joint_angles = np.deg2rad(act_joint_angles)
# rotation = rotation_matrix_to_6d(b2c_rot)  # 旋转矩阵，需要替换
# translation = b2c_trans
# joint_angles_tensor = torch.tensor(joint_angles, dtype=torch.float32).unsqueeze(0).to(device)
# rotation_tensor = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0).to(device)
# translation_tensor = torch.tensor(translation, dtype=torch.float32).unsqueeze(0).to(device)
# keypoints3d = robot.get_keypoints_root(joint_angles_tensor,
#                                 rotation_tensor,
#                                 translation_tensor,
#                                 root=1)

# keypoints3d = keypoints3d.squeeze(0).cpu()
# print(f"keypoints3d: {keypoints3d}")

# project_keypoint_2d = point_projection_from_3d_tensor(torch.FloatTensor(K).unsqueeze(0), keypoints3d.unsqueeze(0))
# print(f"project_keypoint_2d: {project_keypoint_2d}")
# -------------------------------------------

bbox_gt2d = np.concatenate([np.min(keypoints_2d, axis=0), np.max(keypoints_2d, axis=0)])
bbox = get_bbox(bbox_gt2d,w,h,strict=False)

bbox_strict_info = annotations["robot_bounding_box_2d"]
bbox_strict = np.array([bbox_strict_info["x_min"], bbox_strict_info["y_min"], bbox_strict_info["x_max"], bbox_strict_info["y_max"]])
bbox_strict_bounded = np.array([max(0,bbox_strict[0]),max(0,bbox_strict[1]),min(w,bbox_strict[2]),min(h,bbox_strict[3])])
bbox_strict_bounded_original = bbox_strict_bounded.copy()
bbox_strict_bounded_original = torch.FloatTensor(bbox_strict_bounded_original)

def resize_image(image, bbox, K):
    #image as np.array
    wmin, hmin, wmax, hmax = bbox
    square_size =int(max(wmax - wmin, hmax - hmin))
    square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    x_offset = int((square_size - (wmax-wmin)) // 2)
    y_offset = int((square_size- (hmax-hmin)) // 2)
    
    # 使用边界框坐标从原始图像中裁剪出目标区域
    # 将裁剪的区域放置到正方形图像的中心位置，填充到 square_image 中
    square_image[y_offset:y_offset+(hmax-hmin), x_offset:x_offset+(wmax-wmin)] = image[hmin:hmax, wmin:wmax]
        
    K[0, 2] -= (wmin-x_offset)
    K[1, 2] -= (hmin-y_offset)
    
    return square_image, K
# 裁减目标区域调整为正方形并更新相关数据
rgb, K = resize_image(rgb, bbox, K)

def to_torch_uint8(im):
    if isinstance(im, PIL.Image.Image):
        im = torch.as_tensor(np.asarray(im).astype(np.uint8))
    elif isinstance(im, torch.Tensor):
        assert im.dtype == torch.uint8
    elif isinstance(im, np.ndarray):
        assert im.dtype == np.uint8
        im = torch.as_tensor(im)
    else:
        raise ValueError('Type not supported', type(im))
    if im.dim() == 3:
        assert im.shape[-1] in {1, 3},f"{im.shape}"
    return im

def crop(im, K):
    resize = (256, 256)
    im = to_torch_uint8(im)
    
    assert im.shape[-1] == 3
    h, w = im.shape[:2]
    if (h, w) == resize:
        return im, K
    
    images = (torch.as_tensor(im).float() / 255).unsqueeze(0).permute(0, 3, 1, 2)
    K = torch.tensor(K).unsqueeze(0)

    # Resize to target size
    x0, y0 = images.shape[-1] / 2, images.shape[-2] / 2
    h_input, w_input = images.shape[-2], images.shape[-1]
    h_output, w_output = min(resize), max(resize)
    box_size = (h_input, w_input)
    h, w = min(box_size), max(box_size)
    x1, y1, x2, y2 = x0-w/2, y0-h/2, x0+w/2, y0+h/2
    box = torch.tensor([x1, y1, x2, y2])
    images = F.interpolate(images, size=(h_output, w_output), mode='bilinear', align_corners=False)
    K = get_K_crop_resize(K, box.unsqueeze(0), orig_size=(h_input, w_input), crop_resize=(h_output, w_output))
    
    im_np = (images[0].permute(1, 2, 0) * 255).cpu().numpy()
    K = K.squeeze(0).numpy()
    
    return im_np, K

rgb, K = crop(rgb, K)
rgb_np = rgb.copy()
images_r = torch.FloatTensor(rgb.copy()).permute(2,0,1)

# 将原始边界框 bbox_strict_bounded_original 从一个相机坐标系变换到另一个相机坐标系
K_np = K.copy()
K_r = torch.FloatTensor(np.asarray(K))
K_original_inv = np.linalg.inv(K_original)
resize_hw = (256, 256)
bbox_strict_bounded_transformed = bbox_transform(bbox_strict_bounded_original, K_original_inv, np.asarray(K), resize_hw=resize_hw)
bbox_strict_bounded_transformed = np.array([max(0,bbox_strict_bounded_transformed[0]),max(0, bbox_strict_bounded_transformed[1]),
                                            min(resize_hw[0],bbox_strict_bounded_transformed[2]),min(resize_hw[1],bbox_strict_bounded_transformed[3])])
strict_bbox_np = bbox_strict_bounded_transformed.copy()
bbox_strict_bounded_transformed = torch.FloatTensor(bbox_strict_bounded_transformed)

test_image, test_K, test_strict_bbox = images_r.unsqueeze(0), K_r.unsqueeze(0), bbox_strict_bounded_transformed.unsqueeze(0)
display_image, display_strict_bbox = rgb_np, strict_bbox_np

test_ds_names = args.test_ds_names
print(f"using {test_ds_names} for testing")

# path
save_folder = args.exp_path
model_path = os.path.join(save_folder, f"ckpt/{args.model_name}.pk")
result_path = os.path.join(save_folder,  'result')
os.makedirs(result_path, exist_ok=True)

# make models and set initialization 
ckpt = torch.load(model_path) 
init_param_dict = {
    "robot_type" : urdf_robot_name,
    "pose_params": INITIAL_JOINT_ANGLE,
    "cam_params": np.eye(4,dtype=float),
    "init_pose_from_mean": True
}
model = get_rootNetwithRegInt_model(init_param_dict, args)
print("Using rootnet with regression+integral model (2 backbones)")
# Handle loading model trained with DDP (DistributedDataParallel)
state_dict = ckpt['model_state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # Remove "module." prefix if it exists
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
print("This model was saved from epoch:", ckpt["epoch"])

model.eval()

with torch.no_grad():
    real_bbox = torch.tensor([1000.0, 1000.0]).to(torch.float32)
    bboxes = cast(test_strict_bbox, device).float()
    reg_images = cast(test_image.clone(), device).float() / 255.
    root_images = cast(test_image.clone(), device).float() / 255.
    root_K = cast(test_K,device).float()
    fx, fy = root_K[:,0,0], root_K[:,1,1]

    batch_size = 1
    area = torch.max(torch.abs(bboxes[:,2]-bboxes[:,0]), torch.abs(bboxes[:,3]-bboxes[:,1])) ** 2
    k_values = torch.tensor([torch.sqrt(fx[n]*fy[n]*real_bbox[0]*real_bbox[1] / area[n]) for n in range(batch_size)]).to(torch.float32).cuda()

    model.float()
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_id, output_device=device_id[0])

    pred_pose, pred_rot, pred_trans, pred_root_uv, pred_root_depth, \
        pred_uvd, pred_keypoints3d_int, pred_keypoints3d_fk = model(reg_images, root_images, k_values, K=root_K, test_fps=False)
    print(f"pred_pose: {pred_pose}")
    print(f"pred_keypoints3d_int: {pred_keypoints3d_int}")
    print(f"pred_keypoints3d_fk: {pred_keypoints3d_fk}")

import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    # 禁用字体回退警告
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 获取投影的2D关键点
    keypoints3d = pred_keypoints3d_fk.detach().cpu()  # 使用积分预测的3D关键点
    keypoints2d_proj = point_projection_from_3d_tensor(root_K.detach().cpu(), keypoints3d)
    keypoints2d_proj = keypoints2d_proj[0].numpy()  # 只取批次中的第一个样本
    
    # 检查并打印关键点形状
    print(f"关键点形状: {keypoints2d_proj.shape}")
    
    # 绘制图像
    plt.figure(figsize=(10, 10))  # 设置图像大小
    plt.imshow(display_image.astype(np.uint8))
    
    # 绘制边界框
    x_min, y_min, x_max, y_max = display_strict_bbox
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     edgecolor='red', facecolor='none', linewidth=2))
    
    # 绘制投影的2D关键点 - 不使用mask，而是在循环中跳过无效点
    colors = cm.rainbow(np.linspace(0, 1, len(keypoints2d_proj)))
    for i, (point, color) in enumerate(zip(keypoints2d_proj, colors)):
        # 检查点是否有效
        if np.any(np.isnan(point)):
            continue
            
        # 确保坐标是浮点数标量
        x, y = float(point[0]), float(point[1])
        plt.scatter(x, y, color=color, s=50, marker='o')
        plt.text(x + 3, y + 3, str(i), color=color, fontsize=10, weight='bold')
    
    plt.title("带投影关键点的图像")
    plt.axis('off')
    
    # 保存图像到文件
    output_path = 'output_image_with_keypoints.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()  # 关闭图像以释放内存
    
    print(f"带投影关键点的图像已保存至: {output_path}")
    
except Exception as e:
    print(f"可视化过程中出错: {e}")
    import traceback
    traceback.print_exc()