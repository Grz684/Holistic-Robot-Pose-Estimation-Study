import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
from lib.config import LOCAL_DATA_DIR
from lib.core.config import make_cfg
from scripts.train_depthnet import train_depthnet
from scripts.train_sim2real import train_sim2real
from scripts.train_full import train_full
from collections import defaultdict
import numpy as np
import torch
from lib.dataset.const import JOINT_NAMES
from lib.dataset.dream import DreamDataset
from lib.dataset.multiepoch_dataloader import MultiEpochDataLoader
from lib.dataset.samplers import PartialSampler
from lib.models.depth_net import get_rootnet
from lib.utils.urdf_robot import URDFRobot
from lib.utils.utils import cast, set_random_seed, create_logger, get_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    config_path = 'configs/dofbot/depthnet.yaml'
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
    ds_train = DreamDataset(train_ds_names, 
                            # rootnet_resize_hw=(int(cfg.image_size),int(cfg.image_size)), 
                            color_jitter=False, rgb_augmentation=False, occlusion_augmentation=cfg.occlusion, 
                            flip = cfg.rootnet_flip, occlu_p = cfg.occlu_p,
                            padding=cfg.padding, extend_ratio=cfg.extend_ratio)
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
        ds_train, sampler=train_sampler, batch_size=cfg.batch_size, num_workers=cfg.n_dataloader_workers, drop_last=False, pin_memory=True
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
    image_tensor = first_batch["root"]["images"][0]  # Shape应该是 [C, H, W]
    K = first_batch["root"]["K"][0]
    K_original = first_batch["K_original"][0]
    bbox_strict_bounded = first_batch["root"]["bbox_strict_bounded"][0]
    bbox_gt2d_extended = first_batch["root"]["bbox_gt2d_extended"][0]

    print(f"K: {K}, K_original: {K_original}")

    # 打印更详细的图像信息以诊断问题
    print(f"原始图像张量形状: {image_tensor.shape}")
    print(f"原始值范围: min={image_tensor.min().item()}, max={image_tensor.max().item()}")
    print(f"原始图像数据类型: {image_tensor.dtype}")
    
    # 转换为CPU上的numpy数组
    image_np = image_tensor.cpu().detach().numpy()
    
    # 检查通道数并转换
    if image_np.shape[0] == 3 or image_np.shape[0] == 1:  # 如果是CHW格式
        # 转换从[C, H, W]到[H, W, C]
        image_np = image_np.transpose(1, 2, 0)
    
    # 处理单通道图像
    if image_np.shape[-1] == 1:
        image_np = image_np.squeeze(-1)  # 去掉通道维度用于灰度图显示
    
    # 归一化处理
    if image_np.min() < 0 or image_np.max() > 1:
        # 检查是否需要特殊归一化
        if image_np.min() < -0.5 and image_np.max() > 0.5:
            # 可能是[-1,1]范围
            image_np = (image_np + 1) / 2.0
        else:
            # 直接归一化到[0,1]
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
    
    # 确保值在[0,1]范围
    image_np = np.clip(image_np, 0, 1)
    
    print(f"处理后图像形状: {image_np.shape}")
    print(f"处理后值范围: min={image_np.min()}, max={image_np.max()}")
    
    # 显示图像和边界框
    plt.figure(figsize=(10, 10))
    if len(image_np.shape) == 2 or image_np.shape[-1] == 1:  # 灰度图
        plt.imshow(image_np, cmap='gray')
    else:  # RGB图
        plt.imshow(image_np)
    
    # 绘制bbox_strict_bounded - 红色
    bbox_strict = bbox_strict_bounded.cpu().numpy()
    if bbox_strict.size > 0:  # 确保边界框存在
        plt.gca().add_patch(plt.Rectangle((bbox_strict[0], bbox_strict[1]), 
                                         bbox_strict[2] - bbox_strict[0], 
                                         bbox_strict[3] - bbox_strict[1], 
                                         linewidth=2, edgecolor='r', facecolor='none'))
        plt.text(bbox_strict[0], bbox_strict[1] - 5, 'strict_bounded', 
                 color='r', fontsize=12, backgroundcolor='white')
    
    # 绘制bbox_gt2d_extended - 绿色
    bbox_extended = bbox_gt2d_extended.cpu().numpy()
    if bbox_extended.size > 0:  # 确保边界框存在
        plt.gca().add_patch(plt.Rectangle((bbox_extended[0], bbox_extended[1]), 
                                         bbox_extended[2] - bbox_extended[0], 
                                         bbox_extended[3] - bbox_extended[1], 
                                         linewidth=2, edgecolor='g', facecolor='none'))
        plt.text(bbox_extended[0], bbox_extended[1] - 20, 'gt2d_extended', 
                 color='g', fontsize=12, backgroundcolor='white')
    
    plt.axis('off')
    plt.title("样本图像与边界框")
    plt.savefig("sample_image_with_bbox.png")  # 保存图像到文件以便检查
    plt.show()
    
    # 打印边界框信息用于调试
    print(f"bbox_strict_bounded: {bbox_strict_bounded}")
    print(f"bbox_gt2d_extended: {bbox_gt2d_extended}")
    print(first_batch["keypoints_3d_original"][0])
    print(first_batch["jointpose"])

