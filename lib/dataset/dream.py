import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from utils.geometries import quat_to_rotmat_np
from utils.transforms import invert_T
from dataset.augmentations import (CropResizeToAspectAugmentation, FlipAugmentation,
                            PillowBlur, PillowBrightness, PillowColor,
                            PillowContrast, PillowSharpness, occlusion_aug,
                            to_torch_uint8)
from dataset.const import KEYPOINT_NAMES, flip_pairs, rgb_augmentations
from dataset.roboutils import (bbox_transform, get_bbox, get_bbox_raw,
                        get_extended_bbox, make_masks_from_det,
                        process_padding, process_truncation, resize_image,
                        tensor_to_image)


KUKA_SYNT_TRAIN_DR_INCORRECT_IDS = {83114, 28630, }

def build_frame_index(base_dir):
    # im_paths = base_dir.glob('*.jpg')
    im_paths = base_dir.glob('*.png')
    # collections模块中的defaultdict创建一个默认值为列表的字典，用于存储索引信息。
    # 这种数据结构允许直接向不存在的键追加值，而无需预先检查键是否存在。
    infos = defaultdict(list)
    for n, im_path in tqdm(enumerate(sorted(im_paths))):
        # 第二次调用with_suffix('')进一步移除可能存在的第二级扩展名
        view_id = int(im_path.with_suffix('').with_suffix('').name)
        # 特殊情况处理
        if view_id == 0 and "panda_synth_test_photo" in str(base_dir):
            continue
        if 'kuka_synth_train_dr' in str(base_dir) and int(view_id) in KUKA_SYNT_TRAIN_DR_INCORRECT_IDS:
            pass
        else:
            scene_id = view_id
            infos['rgb_path'].append(im_path.as_posix())
            infos['scene_id'].append(scene_id)
            infos['view_id'].append(view_id)
    infos = pd.DataFrame(infos)
    return infos



class DreamDataset(torch.utils.data.Dataset):
    def __init__(self,
                 base_dir,
                 rootnet_resize_hw=(256, 256),
                 other_resize_hw=(256, 256),
                 visibility_check=True,
                 strict_crop=True,
                 color_jitter=True,
                 rgb_augmentation=True,
                 occlusion_augmentation=True,
                 flip=False,
                 rotate=False,
                 padding=False,
                 occlu_p=0.5,
                 process_truncation=False,
                 extend_ratio=[0.2,0.13]
                 ):
        self.base_dir = Path(base_dir)
        self.ds_name = os.path.basename(base_dir)
        self.rootnet_resize_hw=rootnet_resize_hw
        self.other_resize_hw=other_resize_hw
        self.color_jitter=color_jitter
        self.rgb_augmentation=rgb_augmentation
        self.rgb_augmentations=rgb_augmentations
        self.occlusion_augmentation=occlusion_augmentation
        self.total_occlusions = 1
        self.rootnet_flip=flip
        self.rootnet_rotate=rotate
        self.visibility_check = visibility_check
        self.process_truncation = process_truncation
        self.padding = padding
        self.occlu_p = occlu_p
        self.strict_crop = strict_crop
        self.extend_ratio = extend_ratio
    
        self.frame_index = build_frame_index(self.base_dir)
        self.synthetic = True
        if 'panda' in str(base_dir):
            self.keypoint_names = KEYPOINT_NAMES['panda']
            self.label = 'panda'
            if "panda-3cam" in self.ds_name or "panda-orb" in self.ds_name :
                self.synthetic = False
        elif 'baxter' in str(base_dir):
            self.keypoint_names = KEYPOINT_NAMES['baxter']
            self.label = 'baxter'
        elif 'kuka' in str(base_dir):
            self.keypoint_names = KEYPOINT_NAMES['kuka']
            self.label = 'kuka'
        elif 'dofbot' in str(base_dir):
            # KEYPOINT_NAMES指定需要的关键点名称，dofbot取前nkpt个关键点达到了同样的效果
            # self.keypoint_names = KEYPOINT_NAMES['dofbot']
            self.label = 'dofbot'
            pass
        else:
            raise NotImplementedError
        
        self.scale = 0.01 if 'synthetic' in str(self.base_dir) else 1.0
        self.all_labels = [self.label]
        self.flip_pairs=None
        if self.label == 'baxter':
            self.flip_pairs = flip_pairs

        self.nkpt = 8
        self.DOF = 6

    def __len__(self):
        return len(self.frame_index)
    
    def _get_original_and_shared_data(self, idx):
        
        row = self.frame_index.iloc[idx]
        scene_id = row.scene_id
        rgb_path = Path(row.rgb_path)
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
        mask = None
        data_path = rgb_path.with_suffix('').with_suffix('.pkl')
        # annotations = json.loads(rgb_path.with_suffix('').with_suffix('.json').read_text())
        import pickle

        # 加载 pickle 文件
        with open(data_path, 'rb') as f:
            annotations = pickle.load(f)

        # Camera
        TWC = np.eye(4)
        camera_infos_path = self.base_dir / '_camera_settings.json'
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

        camera = dict(
            TWC=TWC,
            resolution=(w, h),
            K=K,
        )
        label = self.label

        # Joints
        # obj_data = annotations['objects'][0]
        # if 'quaternion_xyzw' in obj_data:
        #     rotMat = quat_to_rotmat_np(np.array(obj_data['quaternion_xyzw']))
        #     translation = np.array(obj_data['location']) * self.scale
        #     TWO = np.zeros((4,4), dtype=float)
        #     TWO[:3, :3] = rotMat
        #     TWO[:3, 3] = translation
        #     TWO[3,3] = 1.0
        #     R_NORMAL_UE = np.array([
        #         [0, -1, 0],
        #         [0, 0, -1],
        #         [1, 0, 0],
        #     ])
        #     TWO[:3, :3] = TWO[:3, :3] @ R_NORMAL_UE  
            
        # else:
        #     rotMat = quat_to_rotmat_np(np.array([1.0,0.0,0.0,0.0]))
        #     translation = np.array(obj_data['location']) * self.scale
        #     TWO = np.zeros((4,4), dtype=float)
        #     TWO[:3, :3] = rotMat
        #     TWO[:3, 3] = translation
        #     TWO[3,3] = 1.0
        # TWC = torch.as_tensor(TWC)
        # TWO = torch.as_tensor(TWO)
        # TCO = invert_T(TWC) @ TWO
        # TCO_r = torch.FloatTensor(np.asarray(TCO))

        # TWC中W是上标
        c2b_rot = annotations["camera_ros_axes_to_robot_root_rot"]
        b2c_trans = annotations["keypoint_dict"]["/World/dofbot/link1"]["keypoint_positon"]
        b2c_rot = np.transpose(c2b_rot)
        TCO = np.zeros((4,4), dtype=float)
        TCO[:3, :3] = b2c_rot
        TCO[:3, 3] = b2c_trans
        TCO[3,3] = 1.0
        TCO_r = torch.FloatTensor(np.asarray(TCO))

        # joints = annotations['sim_state']['joints']
        # joints = OrderedDict({d['name'].split('/')[-1]: float(d['position']) for d in joints})
        # if self.label == 'kuka':
        #     joints = {k.replace('iiwa7_', 'iiwa_'): v for k,v in joints.items()}
        joints = annotations["dofbot_joint_names_positions"]

        def normalize_angle(angle):
            # 使用更精确的方法规范化角度
            return np.fmod(angle + 180, 360) - 180

        # 明确使用双精度浮点数
        joints = OrderedDict({k: np.deg2rad(normalize_angle(np.float64(v))) for k, v in joints.items()})
        from dataset.const import JOINT_NAMES
        robot_joints = JOINT_NAMES[self.label]
        joints = OrderedDict({k: joints[k] for k in robot_joints})
        assert len(joints) == self.DOF, f"len(joints)={len(joints)} != {self.DOF}"

        # # keypoints 
        # keypoints_data = obj_data['keypoints']
        # # 代码中的 [None] 是用来增加数组维度的。这是NumPy中常用的一种方法，等同于 np.newaxis。
        # keypoints_2d = np.concatenate([np.array(kp['projected_location'])[None] for kp in keypoints_data], axis=0)
        # # 去重
        # keypoints_2d = np.unique(keypoints_2d, axis=0)
        from lib.dataset.const import LINK_NAMES
        link_names = LINK_NAMES['dofbot']
        keypoints_data = annotations["keypoint_dict"]
        filtered_keypoints_2d = []

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

        # bboxes
        # 基于关键点计算的原始边界框
        bbox_gt2d = np.concatenate([np.min(keypoints_2d, axis=0), np.max(keypoints_2d, axis=0)])
        # bbox用于裁减
        bbox = get_bbox(bbox_gt2d,w,h,strict=self.strict_crop)
        bboxes_r = bbox.copy()
        # bboxs_raw用于有padding的情况
        bboxes_raw = get_bbox_raw(bbox_gt2d)
        bbox_gt2d_extended_original = get_extended_bbox(bbox_gt2d, 20, 20, 20, 20, bounded=True, image_size=(w, h))
        
        # if "bounding_box" in obj_data:
        #     bbox_strict_info = obj_data["bounding_box"]
        #     bbox_strict = np.array([bbox_strict_info["min"][0], bbox_strict_info["min"][1], bbox_strict_info["max"][0], bbox_strict_info["max"][1]])
        #     bbox_strict_bounded = np.array([max(0,bbox_strict[0]),max(0,bbox_strict[1]),min(w,bbox_strict[2]),min(h,bbox_strict[3])])
        if "robot_bounding_box_2d" in annotations:
            bbox_strict_info = annotations["robot_bounding_box_2d"]
            bbox_strict = np.array([bbox_strict_info["x_min"], bbox_strict_info["y_min"], bbox_strict_info["x_max"], bbox_strict_info["y_max"]])
            bbox_strict_bounded = np.array([max(0,bbox_strict[0]),max(0,bbox_strict[1]),min(w,bbox_strict[2]),min(h,bbox_strict[3])])
        else:
            bbox_strict_bounded = bbox_gt2d_extended_original
        bbox_strict_bounded_original = bbox_strict_bounded.copy()
        bbox_gt2d_extended_original = torch.FloatTensor(bbox_gt2d_extended_original)
        bbox_strict_bounded_original = torch.FloatTensor(bbox_strict_bounded_original)
        
        # TCO_keypoints_3d = {kp['name']: np.array(kp['location']) * self.scale for kp in keypoints_data}
        # TCO_keypoints_3d = np.array([TCO_keypoints_3d.get(k, np.nan) for k in self.keypoint_names])
        # assert((np.isnan(TCO_keypoints_3d) == False).all())
        filtered_keypoints_3d = []

        for kp_name, kp_dict in keypoints_data.items():
            # 从格式如 "/World/dofbot/link2" 中提取链接名称
            link_name = kp_name.split('/')[-1]
            
            if link_name in link_names:
                # 这个关键点属于我们感兴趣的链接
                filtered_keypoints_3d.append(np.array(kp_dict['keypoint_positon'])[None])

        # 连接所有过滤后的关键点
        if filtered_keypoints_3d:
            TCO_keypoints_3d = np.concatenate(filtered_keypoints_3d, axis=0)
        else:
            TCO_keypoints_3d = np.empty((0, *np.array(next(iter(keypoints_data.values()))['keypoint_positon']).shape))
        
        assert((np.isnan(TCO_keypoints_3d) == False).all())

        # keypoints_2d = {kp['name']: kp['projected_location'] for kp in keypoints_data}
        # keypoints_2d = np.array([np.append(keypoints_2d.get(k, np.nan) ,0)for k in self.keypoint_names])
        filtered_keypoints_2d = []

        for kp_name, kp_dict in keypoints_data.items():
            # 从格式如 "/World/dofbot/link2" 中提取链接名称
            link_name = kp_name.split('/')[-1]
            
            if link_name in link_names:
                # 这个关键点属于我们感兴趣的链接
                filtered_keypoints_2d.append(np.append(kp_dict['keypoint_projection'], 0))

        # 转换为 numpy 数组
        keypoints_2d = np.array(filtered_keypoints_2d) if filtered_keypoints_2d else np.empty((0, 3))

        # 创建一个二值掩码（binary mask），基于检测框（bounding box）的位置
        mask = make_masks_from_det(bbox[None], h, w).numpy().astype(np.uint8)[0] * 1     
        
        robot = dict(label=label, name=label, joints=joints,
                        # TWO=TWO, 
                        bbox=bbox,
                        id_in_segm=1,
                        keypoints_2d=keypoints_2d,
                        TCO_keypoints_3d=TCO_keypoints_3d)

        # state用于数据增强
        state = dict(
            objects=[robot],
            camera=camera,
            frame_info=row.to_dict()
        )   
        
        K_original = state['camera']['K'].copy()   
        keypoints_2d_original = state['objects'][0]["keypoints_2d"].copy() 
        # 二维关键点的有效性掩码
        valid_mask = (keypoints_2d_original[:, 0] < 640.0) & (keypoints_2d_original[:, 0] >= 0) & \
                     (keypoints_2d_original[:, 1] < 480.0) & (keypoints_2d_original[:, 1] >= 0)
        
        # 这段代码的主要目的是处理物体部分超出图像边界的情况，通过以下步骤解决：
        # 创建更大的图像以容纳完整的物体
        # 将原始图像放置在新图像的适当位置
        # 更新所有相关数据(关键点坐标、相机参数、边界框)以匹配新的图像尺寸和位置
        # if self.process_truncation:
        #     rgb, bbox, mask, state = process_truncation(rgb, bboxes_raw, mask, state)

        # 颜色抖动 (Color Jitter)
        # 40%的概率执行颜色抖动
        # 随机生成颜色因子，创建上下限
        # 对RGB三个通道分别应用不同的随机缩放因子
        # 确保像素值在0-255范围内
        # 将NumPy数组转换回PIL图像
        if self.color_jitter and random.random()<0.4:#color jitter #0.4
            self.color_factor=2*random.random()
            c_high = 1 + self.color_factor
            c_low = 1 - self.color_factor
            rgb=rgb.copy()
            rgb[:, :, 0] = np.clip(rgb[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            rgb[:, :, 1] = np.clip(rgb[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            rgb[:, :, 2] = np.clip(rgb[:, :, 2] * random.uniform(c_low, c_high), 0, 255)
            rgb=Image.fromarray(rgb)
            
        # 遮挡增强 (Occlusion Augmentation)
        # 循环执行指定次数的遮挡增强
        # 每次遮挡有self.occlu_p的概率执行
        # 调用occlusion_aug函数生成遮挡区域的位置和大小
        # 用随机颜色块替换图像中的指定区域（模拟现实世界中的遮挡）
        # 将NumPy数组转换回PIL图像
        for _ in range(self.total_occlusions):
            if self.occlusion_augmentation and random.random() < self.occlu_p: #0.5
                rgb=np.array(rgb)
                synth_ymin, synth_h, synth_xmin, synth_w = occlusion_aug(bbox,np.array([h,w]), min_area=0.0, max_area=0.3, max_try_times=5)
                rgb=rgb.copy()
                rgb[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
                rgb=Image.fromarray(rgb)
            
        # RGB图像增强
        # 创建四种不同类型的图像增强器，每种都有30%的概率被应用：
        # 锐度增强 (Sharpness)：调整图像的锐度
        # 对比度增强 (Contrast)：调整图像的对比度
        # 亮度增强 (Brightness)：调整图像的亮度
        # 颜色增强 (Color)：调整图像的色彩饱和度
        if self.rgb_augmentation :
            # 有些增强类（如CropResizeToAspectAugmentation）会修改obs，例如更新相机参数、关键点坐标等
            # 有些增强类（如PillowSharpness）不需要修改obs，只是为了接口一致性而接收和返回它
            # 这种设计允许所有增强操作使用相同的函数签名（接收图像、掩码和状态，返回处理后的图像、掩码和状态），
            # 这样它们就可以在不需要特殊处理的情况下轻松链式调用或组合使用
            augSharpness = PillowSharpness(p=0.3, factor_interval=(0., 50.)) #0.3
            augContrast = PillowContrast(p=0.3, factor_interval=(0.7, 1.8)) #0.3
            augBrightness = PillowBrightness(p=0.3, factor_interval=(0.7, 1.8)) #0.3
            augColor = PillowColor(p=0.3, factor_interval=(0., 4.)) #0.3
            rgb, mask, state = augSharpness(rgb, mask, state)
            rgb, mask, state = augContrast(rgb, mask, state)
            rgb, mask, state = augBrightness(rgb, mask, state)
            rgb, mask, state = augColor(rgb, mask, state)  
            rgb = np.array(rgb)
            
        keypoints_3d_original = state['objects'][0]["TCO_keypoints_3d"].copy()
        jointpose_r = state["objects"][0]["joints"]
        valid_mask_r = torch.FloatTensor(valid_mask)
        
        # joint是个dict，而keypoints是个list
        return {
                "meta": {"rgb":rgb, "bbox":bbox, "mask":mask, "state":state, "bboxes_raw":bboxes_raw},
                "image_id": idx,
                "scene_id": scene_id,
                "images_original": images_original,
                "bbox_strict_bounded_original": bbox_strict_bounded_original,
                "bbox_gt2d_extended_original": bbox_gt2d_extended_original,
                "TCO": TCO_r,
                "K_original":K_original,
                "jointpose": jointpose_r,
                "keypoints_2d_original":keypoints_2d_original[:,0:2],
                "valid_mask": valid_mask_r,
                "keypoints_3d_original": keypoints_3d_original,
            }
        

        
        
        
    def _get_rootnet_data(self, shared):
        # 数组或字典作为参数传入函数时，函数内部的修改会直接影响原始数据
        # 深拷贝会递归地复制对象及其所有嵌套的子对象，从而生成一个完全独立的新对象
        rgb = deepcopy(shared["meta"]["rgb"])
        bbox = deepcopy(shared["meta"]["bbox"])
        mask = deepcopy(shared["meta"]["mask"])
        state = deepcopy(shared["meta"]["state"])
        bboxes_raw = deepcopy(shared["meta"]["bboxes_raw"])
        K_original = deepcopy(shared["K_original"])
        bbox_strict_bounded_original = deepcopy(shared["bbox_strict_bounded_original"])
        
        if self.rootnet_rotate:
            pass
            # RotationAugmentation
        
        resize_hw = self.rootnet_resize_hw
        rgb = np.asarray(rgb)
        # 裁减目标区域调整为正方形并更新相关数据
        rgb, mask, state = resize_image(rgb, bbox, mask, state)
        # resize图像尺寸
        crop=CropResizeToAspectAugmentation(resize=resize_hw)
        rgb, mask, state = crop(rgb, mask, state)
        if self.rootnet_flip:
            # 对图像及其相关信息（如关键点和相机参数）进行水平翻转操作
            rgb, mask, state = FlipAugmentation(p=0.5,flip_pairs=self.flip_pairs)(rgb, mask, state)
        if self.padding:
            # 填充图像边界
            rgb, bbox, mask, state = process_padding(rgb, bboxes_raw, mask, state, padding_pixel=30)
            rgb, mask, state = crop(rgb, mask, state)
        rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)
        rgb = rgb.permute(2,0,1)
        # rgb变量是一个PIL图像，需要转换为PyTorch张量
        images_r = torch.FloatTensor(np.asarray(rgb))
        K_r = torch.FloatTensor(np.asarray(state['camera']['K']))
        K_original_inv = np.linalg.inv(K_original)
        # 将原始边界框 bbox_strict_bounded_original 从一个相机坐标系变换到另一个相机坐标系
        bbox_strict_bounded_transformed = bbox_transform(bbox_strict_bounded_original, K_original_inv, np.asarray(state['camera']['K']), resize_hw=resize_hw)
        bbox_strict_bounded_transformed = np.array([max(0,bbox_strict_bounded_transformed[0]),max(0, bbox_strict_bounded_transformed[1]),
                                                    min(resize_hw[0],bbox_strict_bounded_transformed[2]),min(resize_hw[1],bbox_strict_bounded_transformed[3])])
        bbox_strict_bounded_transformed = torch.FloatTensor(bbox_strict_bounded_transformed)
        # 根据宽度和高度的扩展比例 self.extend_ratio，对边界框进行扩展（在bbox_gt2d的基础上拓展）
        bbox_from_transformed_gt2d = np.concatenate([np.min(state["objects"][0]["keypoints_2d"], axis=0)[0:2], np.max(state["objects"][0]["keypoints_2d"], axis=0)[0:2]])
        w_, h_ = (bbox_from_transformed_gt2d[2] - bbox_from_transformed_gt2d[0]), (bbox_from_transformed_gt2d[3] - bbox_from_transformed_gt2d[1])
        bbox_gt2d_extended = get_extended_bbox(bbox_from_transformed_gt2d, w_*self.extend_ratio[0], h_*self.extend_ratio[1], 
                                               w_*self.extend_ratio[0], h_*self.extend_ratio[1], bounded=True, image_size=resize_hw)
        bbox_gt2d_extended = torch.FloatTensor(bbox_gt2d_extended)
        
        keypoints_3d_r = torch.FloatTensor(state["objects"][0]["TCO_keypoints_3d"])
        keypoints_2d_r = torch.FloatTensor(state["objects"][0]["keypoints_2d"])[:,0:2]
        keypoints_2d = keypoints_2d_r.numpy()
        valid_mask_crop = torch.FloatTensor( ((keypoints_2d[:, 0] < resize_hw[0]) & (keypoints_2d[:, 0] >= 0) & \
                                            (keypoints_2d[:, 1] < resize_hw[1]) & (keypoints_2d[:, 1] >= 0)))
        
        return {
            "images":images_r, 
            "bbox_strict_bounded": bbox_strict_bounded_transformed, 
            "bbox_gt2d_extended" : bbox_gt2d_extended,
            "K":K_r,
            "keypoints_3d":keypoints_3d_r,
            "keypoints_2d":keypoints_2d_r,
            "valid_mask_crop": valid_mask_crop,
        }
        
    # _get_rootnet_data 使用 self.rootnet_resize_hw 作为调整大小的参数
    # get_other_data 使用 self.other_resize_hw 作为调整大小的参数
    # 除此之外，两个函数并没有什么区别
    def _get_other_data(self, shared):
        rgb = deepcopy(shared["meta"]["rgb"])
        bbox = deepcopy(shared["meta"]["bbox"])
        mask = deepcopy(shared["meta"]["mask"])
        state = deepcopy(shared["meta"]["state"])
        bboxes_raw = deepcopy(shared["meta"]["bboxes_raw"])
        K_original = deepcopy(shared["K_original"])
        bbox_strict_bounded_original = deepcopy(shared["bbox_strict_bounded_original"])
        
        resize_hw = self.other_resize_hw
        rgb = np.asarray(rgb)
        rgb, mask, state = resize_image(rgb, bbox, mask, state)

        crop=CropResizeToAspectAugmentation(resize=resize_hw)
        rgb, mask, state = crop(rgb, mask, state)
        if self.padding:
            rgb, bbox, mask, state = process_padding(rgb, bboxes_raw, mask, state, padding_pixel=30)
            rgb, mask, state = crop(rgb, mask, state)
        rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)
        rgb = rgb.permute(2,0,1)

        images_r = torch.FloatTensor(np.asarray(rgb))
        K_r = torch.FloatTensor(np.asarray(state['camera']['K']))
        K_original_inv = np.linalg.inv(K_original)
        bbox_strict_bounded_transformed = bbox_transform(bbox_strict_bounded_original, K_original_inv, np.asarray(state['camera']['K']), resize_hw=resize_hw)
        bbox_strict_bounded_transformed = np.array([max(0,bbox_strict_bounded_transformed[0]),max(0, bbox_strict_bounded_transformed[1]),
                                                    min(resize_hw[0],bbox_strict_bounded_transformed[2]),min(resize_hw[1],bbox_strict_bounded_transformed[3])])
        bbox_strict_bounded_transformed = torch.FloatTensor(bbox_strict_bounded_transformed)

        bbox_from_transformed_gt2d = np.concatenate([np.min(state["objects"][0]["keypoints_2d"], axis=0)[0:2], np.max(state["objects"][0]["keypoints_2d"], axis=0)[0:2]])
        w_, h_ = (bbox_from_transformed_gt2d[2] - bbox_from_transformed_gt2d[0]), (bbox_from_transformed_gt2d[3] - bbox_from_transformed_gt2d[1])
        bbox_gt2d_extended = get_extended_bbox(bbox_from_transformed_gt2d, w_*self.extend_ratio[0], h_*self.extend_ratio[1], w_*self.extend_ratio[0], h_*self.extend_ratio[1], bounded=True, image_size=resize_hw)
        bbox_gt2d_extended = torch.FloatTensor(bbox_gt2d_extended)
        
        keypoints_3d_r = torch.FloatTensor(state["objects"][0]["TCO_keypoints_3d"])
        keypoints_2d_r = torch.FloatTensor(state["objects"][0]["keypoints_2d"])[:,0:2]
        keypoints_2d = keypoints_2d_r.numpy()
        valid_mask_crop = torch.FloatTensor( ((keypoints_2d[:, 0] < resize_hw[0]) & (keypoints_2d[:, 0] >= 0) & \
                                            (keypoints_2d[:, 1] < resize_hw[1]) & (keypoints_2d[:, 1] >= 0)))
        
        return {
            "images":images_r, 
            "bbox_strict_bounded": bbox_strict_bounded_transformed, 
            "bbox_gt2d_extended" : bbox_gt2d_extended,
            "K":K_r,
            "keypoints_3d":keypoints_3d_r,
            "keypoints_2d":keypoints_2d_r,
            "valid_mask_crop": valid_mask_crop,
        }
        
        
        
    
    def __getitem__(self, idx):
        
        shared_data = self._get_original_and_shared_data(idx)
        rootnet_data = self._get_rootnet_data(shared_data)
        other_data = self._get_other_data(shared_data)
        
        return {
                "image_id": shared_data["image_id"],
                "scene_id": shared_data["scene_id"],
                "images_original": shared_data["images_original"],
                "bbox_strict_bounded_original": shared_data["bbox_strict_bounded_original"],
                "bbox_gt2d_extended_original": shared_data["bbox_gt2d_extended_original"],
                "TCO":shared_data["TCO"],
                "K_original":shared_data["K_original"],
                "jointpose":shared_data["jointpose"],
                "keypoints_2d_original":shared_data["keypoints_2d_original"],
                "valid_mask": shared_data["valid_mask"],
                "keypoints_3d_original": shared_data["keypoints_3d_original"],
                "root": rootnet_data,
                "other": other_data
            }