import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 只让程序看到第4个GPU

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from lib.core.config import make_cfg
import numpy as np
import torch
from lib.dataset.const import JOINT_NAMES
from lib.dataset.dream import DreamDataset
from lib.dataset.samplers import PartialSampler
from lib.utils.urdf_robot import URDFRobot
from lib.utils.utils import cast, set_random_seed, create_logger, get_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from lib.models.full_net import get_rootNetwithRegInt_model
from lib.dataset.const import INITIAL_JOINT_ANGLE
from collections import OrderedDict
from lib.utils.transforms import point_projection_from_3d_tensor
from lib.utils.geometries import (
    angle_axis_to_rotation_matrix, compute_geodesic_distance_from_two_matrices,
    quat_to_rotmat, rot6d_to_rotmat, rotmat_to_quat, rotmat_to_rot6d)
    
parser = argparse.ArgumentParser('Training')
config_path = 'configs/dofbot/load_dataset.yaml'
parser.set_defaults(config=config_path)
args = parser.parse_args()
cfg = make_cfg(args)

print("-------------------   config for this experiment   -------------------")
print(cfg)
print("----------------------------------------------------------------------")

urdf_robot_name = cfg.urdf_robot_name
robot = URDFRobot(urdf_robot_name)

device_id = cfg.device_id
device = torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")

train_ds_names = cfg.train_ds_names
test_ds_name_dr = train_ds_names.replace("train_dr","test_dr")
if urdf_robot_name != "baxter" and urdf_robot_name != "dofbot":
    test_ds_name_photo = train_ds_names.replace("train_dr","test_photo")
if urdf_robot_name == "panda":
    test_ds_name_real = [train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-3cam_azure"),
                        train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-3cam_kinect360"),
                        train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-3cam_realsense"),
                        train_ds_names.replace("synthetic/panda_synth_train_dr","real/panda-orb")]
    
# make train and test(validation) datasets/dataloaders
ds_train = DreamDataset(
    train_ds_names,
    rootnet_resize_hw=(256, 256), 
    other_resize_hw=(256, 256),
    color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False
)
# ds_test_dr = DreamDataset(test_ds_name_dr, 
#                           # rootnet_resize_hw=(int(cfg.image_size),int(cfg.image_size)), 
#                           color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False, 
#                           flip = False,
#                           padding=cfg.padding, extend_ratio=cfg.extend_ratio) 
# if urdf_robot_name != "baxter" and urdf_robot_name != "dofbot":
#     ds_test_photo = DreamDataset(test_ds_name_photo, 
#                                  # rootnet_resize_hw=(int(cfg.image_size),int(cfg.image_size)), 
#                                  color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False, 
#                                  flip = False,
#                                  padding=cfg.padding, extend_ratio=cfg.extend_ratio) 
# 代码实现了从训练数据集中随机采样指定数量的数据项，以便在每个 epoch 中使用不同的子集进行训练。这种方法有助于提高训练的多样性和模型的泛化能力。
train_sampler = PartialSampler(ds_train, epoch_size=cfg.epoch_size)
if cfg.resample:
    # 采样器（Sampler）是 PyTorch 数据加载器（DataLoader）中的一个组件，用于定义从数据集中抽取样本的策略。采样器决定了数据加载器在每个 epoch 中如何遍历数据集
    # WeightedRandomSampler：根据给定的权重随机抽取样本，权重较大的样本被抽取的概率较高。
    weights_sampler = np.load("unit_test/z_weights.npy")
    train_sampler = WeightedRandomSampler(weights_sampler, num_samples=min(cfg.epoch_size, len(ds_train))) 
ds_iter_train = DataLoader(
    ds_train, sampler=train_sampler, batch_size=1, num_workers=cfg.n_dataloader_workers, drop_last=False, pin_memory=True
)
# ds_iter_train = MultiEpochDataLoader(ds_iter_train)

# test_loader_dict = {}
# ds_iter_test_dr = DataLoader(
#     ds_test_dr, batch_size=cfg.batch_size, num_workers=cfg.n_dataloader_workers
# )
# test_loader_dict["dr"] = ds_iter_test_dr
# if urdf_robot_name != "baxter" and urdf_robot_name != "dofbot":
#     ds_iter_test_photo = DataLoader(
#         ds_test_photo, batch_size=cfg.batch_size, num_workers=cfg.n_dataloader_workers
#     )

# if urdf_robot_name == "panda":
#     ds_shorts = ["azure", "kinect", "realsense", "orb"]
#     for ds_name, ds_short in zip(test_ds_name_real, ds_shorts):
#         ds_test_real = DreamDataset(ds_name, 
#                                     # rootnet_resize_hw=(int(cfg.image_size),int(cfg.image_size)), 
#                                     color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False, 
#                                     flip = False,
#                                     padding=cfg.padding, extend_ratio=cfg.extend_ratio) 
#         ds_iter_test_real = DataLoader(
#             ds_test_real, batch_size=cfg.batch_size, num_workers=cfg.n_dataloader_workers
#         )
#         # test_loader_dict字典保存多个真实数据集的DataLoader
#         test_loader_dict[ds_short] = ds_iter_test_real

print("len(ds_iter_train): ",len(ds_iter_train))
# print("len(ds_iter_test_dr): ", len(ds_iter_test_dr))
# if urdf_robot_name != "baxter" and urdf_robot_name != "dofbot":
#     print("len(ds_iter_test_photo): ", len(ds_iter_test_photo))
# if urdf_robot_name == "panda":
#     for ds_short in ds_shorts:
#         print(f"len(ds_iter_test_{ds_short}): ", len(test_loader_dict[ds_short]))

# Retrieve the first value from ds_iter_train

first_batch = next(iter(ds_iter_train))
# first_sample = first_batch[0]  # 假设 batch 是一个列表或字典
print("First batch type:", type(first_batch))
print(first_batch.keys())
# Display the first image in the batch
import matplotlib.pyplot as plt

# 获取第一张图片
root_images = cast(first_batch["root"]["images"],device).float() / 255.
root_K = cast(first_batch["root"]["K"],device).float()

reg_images = cast(first_batch["other"]["images"],device).float() / 255.
other_K = cast(first_batch["other"]["K"],device).float()

TCO = cast(first_batch["TCO"],device).float()
K_original = cast(first_batch["K_original"],device).float()
gt_jointpose = first_batch["jointpose"]
gt_keypoints2d_original = cast(first_batch["keypoints_2d_original"],device).float()
valid_mask = cast(first_batch["valid_mask"],device).float()
gt_keypoints2d = cast(first_batch["other"]["keypoints_2d"],device).float()
valid_mask_crop = cast(first_batch["other"]["valid_mask_crop"],device).float()
gt_keypoints3d = cast(first_batch["other"]["keypoints_3d"],device).float()

# 边界框选择与批次大小确定
batch_size = root_images.shape[0]
robot_type = cfg.urdf_robot_name
bboxes = cast(first_batch["root"]["bbox_strict_bounded"], device).float()
gt_pose = []
gt_rot = []
gt_trans = []
for n in range(batch_size):
    # 为每个样本构建关节姿态、旋转和平移的真实值
    jointpose = torch.as_tensor([gt_jointpose[k][n] for k in JOINT_NAMES[robot_type]])
    jointpose = cast(jointpose, device,dtype=torch.float)
    rot6d = rotmat_to_rot6d(TCO[n,:3,:3])
    trans = TCO[n,:3,3]
    gt_pose.append(jointpose)
    gt_rot.append(rot6d)
    gt_trans.append(trans)
gt_pose = torch.stack(gt_pose, 0).to(torch.float32)
gt_rot = torch.stack(gt_rot, 0).to(torch.float32)
if cfg.rotation_dim == 4:
    gt_rot = rotmat_to_quat(TCO[:,:3,:3])
gt_trans = torch.stack(gt_trans, 0).to(torch.float32)

print((f"gt_pose: {gt_pose}, gt_rot: {gt_rot}, gt_trans: {gt_trans}"))

print(f"真实值的3D关键点: {gt_keypoints3d}")

a = robot.get_keypoints_root(gt_pose, gt_rot, gt_trans, 1)
print(f"运动学计算的3D关键点: {a}")

test_image, test_K, test_strict_bbox = root_images, root_K, bboxes

# 假设root_images是一个形状为[batch_size, channels, height, width]的tensor
# 获取第一张图片并转换为numpy数组
display_image = root_images[0].cpu().detach().numpy()

# 如果需要将通道维度从第一维移到最后一维(从CHW转为HWC格式)
# 比如从[channels, height, width]转为[height, width, channels]
# 这在显示RGB图像时通常需要
if len(display_image.shape) == 3:  # 如果是3维数组(有通道维度)
    display_image = np.transpose(display_image, (1, 2, 0))

display_strict_bbox = bboxes[0].cpu().detach().numpy()

# path
save_folder = "experiments/dofbot_full_not_direct_reg/"
model_path = os.path.join(save_folder, f"ckpt/curr_best_auc(add)_model.pk")
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
model = get_rootNetwithRegInt_model(init_param_dict, cfg)
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
    bboxes = test_strict_bbox
    reg_images = test_image
    root_images = test_image
    root_K = test_K
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
    print(f"pred_rot: {pred_rot}")
    print(f"pred_trans: {pred_trans}")
    print(f"pred_keypoints3d_int: {pred_keypoints3d_int}")
    print(f"pred_keypoints3d_fk: {pred_keypoints3d_fk}")

import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    # 禁用字体回退警告
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 获取投影的2D关键点
    keypoints3d = pred_keypoints3d_int.detach().cpu()  # 使用积分预测的3D关键点
    keypoints2d_proj = point_projection_from_3d_tensor(root_K.detach().cpu(), keypoints3d)
    keypoints2d_proj = keypoints2d_proj[0].numpy()  # 只取批次中的第一个样本
    
    # 检查并打印关键点形状
    print(f"关键点形状: {keypoints2d_proj.shape}")
    
    # 绘制图像
    plt.figure(figsize=(10, 10))  # 设置图像大小
    plt.imshow(display_image)
    
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