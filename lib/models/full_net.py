import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import time
import numpy as np
import torch
import torch.nn as nn
from dataset.const import JOINT_BOUNDS, JOINT_NAMES
from .backbones.HRnet import get_hrnet
from .backbones.Resnet import get_resnet
from utils.geometries import rot6d_to_rotmat, rotmat_to_quat, rotmat_to_rot6d
from utils.integral import HeatmapIntegralJoint, HeatmapIntegralPose
from utils.transforms import uvz2xyz_singlepoint
from utils.urdf_robot import URDFRobot

class RootNetwithRegInt(nn.Module):
    """ RootNetwithRegInt model with ResNet backbone

        In previous works' terminology:
        pose refers to the rotation configuration of the human keypoints (npose = 24*6)
        cam refers to the human-to-camera transformation

        In our work: (in order to be consistent with previous works such as HMR)
        pose refers to the configuration(angle) values of the robot joints  (npose = DoF)
        cam refers to the robot-to-camera 6D pose/transformation,
        which is presented in rotation(dim=6 representation, Zhou et al) and translation(dim=3)

        initialization parameters dict {
            "robot_type" : str
            "pose_params": dict
            "cam_params": array(float) (as in 4x4 array)
            "init_pose_from_mean": bool
        }
    """

    def __init__(self, init_param_dict, args, **kwargs):

        super(RootNetwithRegInt, self).__init__()

        robot_type = init_param_dict["robot_type"]
        if robot_type == "panda":
            DoF = 8
            nkpt = 7
        elif robot_type == "kuka":
            DoF = 7
            nkpt = 8
        elif robot_type == "baxter":
            DoF = 15
            nkpt = 17
        elif robot_type == "dofbot":
            DoF = 6
            nkpt = 8
        else:
            raise ValueError(f"Robot type {robot_type} is not supported.")
        npose = DoF
        self.robot = URDFRobot(robot_type)
        self.backbone_name = args.backbone_name
        self.rootnet_backbone_name = args.rootnet_backbone_name 
        self.use_rpmg = args.use_rpmg
        self.n_iter = args.n_iter
        self.norm_type = "softmax"
        self.deconv_dim = [256,256,256]
        self.num_joints = nkpt
        self.image_size = args.other_image_size
        self.depth_dim = 64
        self.height_dim = int(self.image_size/4)
        self.width_dim = int(self.image_size/4)
        self.bbox_3d_shape = args.bbox_3d_shape
        self.reference_keypoint_id = args.reference_keypoint_id
        self.integral_layer = HeatmapIntegralPose(backbone=self.backbone_name, num_joints=self.num_joints, depth_dim=self.depth_dim,
                                                  height_dim=self.height_dim, width_dim=self.width_dim, norm_type=self.norm_type,
                                                  image_size=self.image_size, bbox_3d_shape=self.bbox_3d_shape, rootid=self.reference_keypoint_id,
                                                  fixroot=args.fix_root)
        self.rotation_dim = args.rotation_dim
        # JointNet\RotationNet\KeypoinNet的共享backbone
        if self.backbone_name in ["resnet", "resnet34", "resnet50", "resnet101"]:
            self.reg_backbone = get_resnet(self.backbone_name)
            self.feature_channel = self.reg_backbone.block.expansion * 512
            self.deconv_layers = self._make_deconv_layer()
            self.final_layer = nn.Conv2d(self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)
            self.avgpool = nn.AvgPool2d(int(self.image_size/32), stride=1)
        elif self.backbone_name in ["hrnet", "hrnet32"]:
            self.reg_backbone = get_hrnet(type_name=32, num_joints=self.num_joints, depth_dim=self.depth_dim,
                                      pretrain=True, generate_feat=True, generate_hm=True)
            self.feature_channel = 2048
        else:
            raise(NotImplementedError)
        
        # npose与DoF对应
        self.reg_joint_map = args.reg_joint_map
        if self.reg_joint_map:
            self.joint_conv_dim = args.joint_conv_dim
            self.joint_conv_layers = self._make_joint_conv_layer()
            self.joint_final_layer = nn.Conv2d(self.joint_conv_dim[2], npose, kernel_size=1, stride=1, padding=0)
            joint_bounds = torch.tensor(JOINT_BOUNDS[robot_type]).float()
            self.joint_integral_layer = HeatmapIntegralJoint(backbone=self.backbone_name, dof=npose, norm_type=self.norm_type, joint_bounds=joint_bounds)
        else:
            self.fc_pose_1 = nn.Linear(self.feature_channel + npose, 1024)
            self.fc_pose_2 = nn.Linear(1024, 1024)
            self.decpose = nn.Linear(1024, npose)
            self.drop1 = nn.Dropout(p=args.p_dropout)
            self.drop2 = nn.Dropout(p=args.p_dropout)
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        
        self.direct_reg_rot = args.direct_reg_rot
        self.rot_iterative_matmul = args.rot_iterative_matmul
        # self.rot_dim = args.rot_dim
        if self.direct_reg_rot:
            self.fc_rot_1 = nn.Linear(self.feature_channel, 1024)
            self.fc_rot_2 = nn.Linear(1024, 1024)
            self.fc_rot_3 = nn.Linear(1024, 1024)
            self.fc_rot_4 = nn.Linear(1024, 1024)
            self.fc_rot_5 = nn.Linear(1024, 1024)
            self.fc_rot_6 = nn.Linear(1024, 1024)
            self.decrot = nn.Linear(1024, 6)
            self.drop_rot = nn.Dropout(p=0.2)
            # self.fc_rot_1 = nn.Linear(self.feature_channel, 1024)
            # self.bn_rot_1 = nn.BatchNorm1d(1024)
            # self.fc_rot_2 = nn.Linear(1024, 512) 
            # self.bn_rot_2 = nn.BatchNorm1d(512)
            # self.fc_rot_3 = nn.Linear(512, 256) 
            # self.bn_rot_3 = nn.BatchNorm1d(256)
            # self.fc_rot_4 = nn.Linear(256, 128) 
            # self.bn_rot_4 = nn.BatchNorm1d(128)
            # self.fc_rot_5 = nn.Linear(128, 32) 
            # self.bn_rot_5 = nn.BatchNorm1d(32)
            # self.relu_rot = nn.ReLU()
            # self.decrot = nn.Linear(32, 6)
            # self.fc_rot_05 = nn.Linear(self.feature_channel, 32) 
            nn.init.xavier_uniform_(self.decrot.weight, gain=0.01)
        else:
            self.fc_rot_1 = nn.Linear(self.feature_channel + self.rotation_dim, 1024)
            self.fc_rot_2 = nn.Linear(1024, 1024) 
            self.decrot = nn.Linear(1024, self.rotation_dim) # rotation representation (quaternion / 6D / 9D)
            self.drop1 = nn.Dropout(p=args.p_dropout)
            self.drop2 = nn.Dropout(p=args.p_dropout)
            nn.init.xavier_uniform_(self.decrot.weight, gain=0.01)

        # depthnet的backbone
        if self.rootnet_backbone_name in ["resnet",  "resnet50",  "resnet34"]:
            self.rootnet_backbone = get_resnet(self.rootnet_backbone_name)
            self.inplanes = self.rootnet_backbone.block.expansion * 512
        elif self.rootnet_backbone_name in ["hrnet",  "hrnet32"]:
            # 当generate_hm=False且generate_feat=True时：只输出特征向量，形状为[batch_size, 2048]
            # hm是heatmap，feat是特征向量
            self.rootnet_backbone = get_hrnet(type_name=32, num_joints=nkpt, depth_dim=self.depth_dim,
                                            pretrain=True, generate_feat=True, generate_hm=False)
            self.inplanes = 2048
        else:
            raise(NotImplementedError)
        
        self.multi_kp = args.multi_kp
        self.kps_need_depth = args.kps_need_depth if self.multi_kp else [args.reference_keypoint_id]
        self.depth_num = len(self.kps_need_depth)
        self.add_fc = args.add_fc

        # 启用额外的全连接层处理
        if self.add_fc:
            self.depth_dropout = nn.Dropout(p=0.2)
            self.depth_fc_d1 = nn.Linear(self.inplanes, 1024)
            self.depth_fc_d2 = nn.Linear(1024, 512)
            self.depth_bn = nn.BatchNorm1d(512)
            self.depth_lrelu = nn.LeakyReLU()
            self.depth_fc_u2 = nn.Linear(512, 1024)
            self.depth_fc_u1 = nn.Linear(1024, self.inplanes)
            
        # self.depth_num 个不同的卷积核，每个卷积核的尺寸是 [self.inplanes, 1, 1]，
        # 它对输入的每个通道应用不同的权重，然后将它们求和生成输出通道。
        self.depth_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=self.depth_num, 
            kernel_size=1,
            stride=1,
            padding=0
        )  

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for m in self.depth_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        init_param_pose = init_param_dict["pose_params"]
        init_param_cam = init_param_dict["cam_params"]
        init_param_from_mean = init_param_dict["init_pose_from_mean"]
        # init_param_dict = {
        #     "robot_type" : urdf_robot_name,
        #     "pose_params": INITIAL_JOINT_ANGLE,
        #     "cam_params": np.eye(4,dtype=float),
        #     "init_pose_from_mean": True
        # }
        # np.eye(4,dtype=float) 表示使用一个恒等变换
        if init_param_from_mean:
            init_pose = torch.from_numpy(np.array([init_param_pose['mean'][robot_type][k] for k in JOINT_NAMES[robot_type]])).unsqueeze(0).float()
        else:
            init_pose = torch.from_numpy(np.array([init_param_pose['zero'][robot_type][k] for k in JOINT_NAMES[robot_type]])).unsqueeze(0).float()
        if self.rotation_dim == 6:
            init_rot = rotmat_to_rot6d(torch.from_numpy(np.array(init_param_cam[:3,:3])).unsqueeze(0)).float()
        elif self.rotation_dim == 4:
            init_rot = rotmat_to_quat(torch.from_numpy(np.array(init_param_cam[:3,:3])).unsqueeze(0)).float()
        
        # 缓冲区是模型的一部分，但不会被视为模型的参数，因此不会在训练过程中更新。它们通常用于存储固定的值
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_rot', init_rot)

    def _make_deconv_layer(self):
        deconv_layers = []
        # 每层转置卷积操作都将空间维度扩大一倍，同时可能减少通道数，
        # 逐步将深层特征转换为空间分辨率更高的特征图。
        deconv1 = nn.ConvTranspose2d(
            self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = nn.BatchNorm2d(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = nn.BatchNorm2d(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = nn.BatchNorm2d(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)
    
    def _make_joint_conv_layer(self):
        joint_conv_layers = []
        conv1 = nn.Conv2d(self.feature_channel, self.joint_conv_dim[0], kernel_size=3, stride=1, padding=1)
        bn1 = nn.BatchNorm2d(self.joint_conv_dim[0])
        conv2 = nn.Conv2d(self.joint_conv_dim[0], self.joint_conv_dim[1], kernel_size=3, stride=1, padding=1)
        bn2 = nn.BatchNorm2d(self.joint_conv_dim[1])
        conv3 = nn.Conv2d(self.joint_conv_dim[1], self.joint_conv_dim[2], kernel_size=3, stride=1, padding=1)
        bn3 = nn.BatchNorm2d(self.joint_conv_dim[2])
        
        joint_conv_layers.append(conv1)
        joint_conv_layers.append(bn1)
        joint_conv_layers.append(nn.ReLU(inplace=True))
        joint_conv_layers.append(conv2)
        joint_conv_layers.append(bn2)
        joint_conv_layers.append(nn.ReLU(inplace=True))
        joint_conv_layers.append(conv3)
        joint_conv_layers.append(bn3)
        joint_conv_layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*joint_conv_layers)

    # 两个骨干网络的前向传播是独立进行的，但它们的输出会在后续的计算中结合起来，以生成最终的预测结果
    def forward(self, x_reg_input, x_root_input, k_value, K, init_pose=None, init_rot=None, test_fps=False):

        batch_size = x_reg_input.shape[0]
        x_reg_input = x_reg_input.to(torch.float)
        x_root_input = x_root_input.to(torch.float)

        if init_pose is None:
            # 拓展到batch_size（不过这里self.init_pose好像没定义）
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_rot is None:
            init_rot = self.init_rot.expand(batch_size, -1)
        root_trans_from_rootnet = torch.zeros((batch_size, 3)).float()

        # root z
        if self.rootnet_backbone_name in ["resnet34", "resnet50", "resnet", "resnet101"]:
            if test_fps:
                t_start_root = time.time()
            fm = self.rootnet_backbone(x_root_input)
            # Tensor 对象的一个方法，用于改变张量的形状（即重塑张量）。类似于reshape 方法

            # 全局平均池化：将空间维度压缩为1个值
            # 首先将feature map重塑为[batch_size, channels, height*width]
            # 然后在dim=2上取平均，得到[batch_size, channels]
            img_feat = torch.mean(fm.view(fm.size(0), fm.size(1), fm.size(2)*fm.size(3)), dim=2) # global average pooling
        elif self.rootnet_backbone_name in ["hrnet", "hrnet32"]:
            if test_fps:
                t_start_root = time.time()
            img_feat = self.rootnet_backbone(x_root_input)
        # 特征增强（添加全连接层）
        # 实现了一个类似U-Net结构的特征处理模块：先通过全连接层将特征降维，然后再通过全连接层将特征恢复，
        # 并添加了跳跃连接。这种结构有助于保留细节信息并使模型更容易训练。
        if self.add_fc:
            # 下采样路径：特征降维
            img_feat1 = self.depth_fc_d1(img_feat) # [batch_size, 1024]
            img_feat2 = self.depth_fc_d2(img_feat1) # [batch_size, 512]

            # 规范化和激活
            img_feat_mid = self.depth_bn(img_feat2)
            img_feat_mid = self.depth_lrelu(img_feat_mid)

            # 上采样路径：特征重建
            img_feat3 = self.depth_fc_u2(img_feat_mid) # [batch_size, 1024]
            # 跳跃连接：将上采样特征与下采样特征相加
            img_feat3 = 0.5 * (img_feat3 + img_feat1)

            img_feat4 = self.depth_fc_u1(img_feat3) # [batch_size, original_channels]
            # 跳跃连接：将上采样特征与原始特征相加
            img_feat4 = 0.5 * (img_feat4 + img_feat)
            img_feat = img_feat4 

        img_feat = torch.unsqueeze(img_feat,2)
        img_feat = torch.unsqueeze(img_feat,3)

        gamma = self.depth_layer(img_feat) # [batch_size, depth_num, 1, 1]
        gamma = gamma.view(-1,1)
        # 多关键点深度预测
        if self.multi_kp:
            # 将gamma调整为[batch_size, depth_num]并乘以比例因子
            # k = sqrt((fx * fy * real_width * real_height) / image_area)，用来从2D图像的尺寸推算3D绝对深度
            # image_area是训练输入值的bbox信息计算得到的（每个输入不同？那预测时也需要这个输入值吗？）
            # real_bbox = torch.tensor([1000.0, 1000.0]).to(torch.float32)似乎是个固定值
            # 机器人可以在图像中呈现相同深度但不同大小的姿势，这使得dc不足以准确表示根部深度。校正因子λ（即这里的gamma）为深度估计引入了非线性灵活性，通过基于图像特征调整dc来解决这个问题。
            # expand作用为在第二维度上复制扩展到 self.depth_num
            pred_depths = gamma.view(-1,self.depth_num) * k_value.view(-1,1).expand(-1, self.depth_num)
            # 将单位从毫米转换为米
            pred_depths = pred_depths / 1000.0
            # 获取参考关键点的深度
            root_index = self.kps_need_depth.index(self.reference_keypoint_id)
            pred_depth = pred_depths[:,root_index].reshape(-1,1)    
        # 单关键点深度预测
        else:
            # 直接将gamma乘以比例因子k_value
            pred_depth = gamma * k_value.view(-1,1)
            # 调整形状并转换单位
            pred_depth = pred_depth.reshape(img_feat.size(0), 1) / 1000.0
        if test_fps:
            torch.cuda.current_stream().synchronize()
            t_end_root = time.time()
            time_root = t_end_root - t_start_root 
        # 得到预测深度
        root_trans_from_rootnet[:,2:3] = pred_depth
        
        if test_fps:
            t_start_other = time.time() 
        # integral uvd xyz
        if self.backbone_name in ["resnet", "resnet50", "resnet34"]:
            x_out = self.reg_backbone(x_reg_input)
            # 原始特征 x 的一种全局特征表示形式xf
            xf = self.avgpool(x_out)

            # 上采样和热图生成,空间维度扩大8倍
            out = self.deconv_layers(x_out)
            # 输出channel数为num_joints * depth_dim（B×[N×D]×H′×W ′）
            out = self.final_layer(out)
            # K是相机内参矩阵
            # 由于图片尺寸是256×256，resnet缩小为1/32，这里上采样放大8倍，所以H′=W′=64
            # integral_layer中设定H′,W′为image size的1/4，不受图片大小的影响
            pred_uvd, pred_xyz_int = self.integral_layer(out, root_trans=root_trans_from_rootnet, K=K)
            pred_root_uv = (pred_uvd[:,self.reference_keypoint_id,:2]+ 0.5) * self.image_size
        elif self.backbone_name in ["hrnet", "hrnet32"]:
            out, xf = self.reg_backbone(x_reg_input)
            pred_uvd, pred_xyz_int = self.integral_layer(out, root_trans=root_trans_from_rootnet, K=K)
            pred_root_uv = (pred_uvd[:,self.reference_keypoint_id,:2] + 0.5) * self.image_size

        # root trans (xyz)
        pred_trans = uvz2xyz_singlepoint(pred_root_uv, pred_depth, K)
        
        # joint angle/pose, rotation (iterative)
        pred_pose = init_pose
        pred_rot = init_rot
        # reshape为一个向量(batch size)
        xf = xf.view(xf.size(0), -1)
        
        # skiplist, skiplist2 = {}, {}
        if self.reg_joint_map:
            joint_out = self.joint_conv_layers(x_out)
            joint_out = self.joint_final_layer(joint_out)
            pred_pose = self.joint_integral_layer(joint_out)
        else:
            for i in range(self.n_iter):
                xc = torch.cat([xf, pred_pose],1).to(torch.float)
                xc = self.fc_pose_1(xc)
                xc = self.drop1(xc)
                # if i < int(self.n_iter / 2):
                #     skiplist[i] = xc
                # elif i == int(self.n_iter / 2) and self.n_iter % 2 != 0:
                #     pass
                # else:
                #     xplus = skiplist[i-int((self.n_iter+1) / 2)]
                #     xc += xplus
                xc = self.fc_pose_2(xc)
                xc = self.drop2(xc)
                pred_pose = self.decpose(xc) + pred_pose
                
        if self.direct_reg_rot:
            # xc = self.relu_rot(self.bn_rot_1(self.fc_rot_1(xf)))
            # xc = self.relu_rot(self.bn_rot_2(self.fc_rot_2(xc)))
            # xc = self.relu_rot(self.bn_rot_3(self.fc_rot_3(xc)))
            # xc = self.relu_rot(self.bn_rot_4(self.fc_rot_4(xc)))
            # xc = self.relu_rot(self.bn_rot_5(self.fc_rot_5(xc) + self.fc_rot_05(xf)))
            xc1 = self.fc_rot_1(xf)
            xc2 = self.fc_rot_2(xc1)
            xc3 = self.fc_rot_3(xc2)
            xc4 = self.fc_rot_4(xc3)
            xc5 = self.fc_rot_5(xc4)
            xc6 = self.fc_rot_6(xc5)
            xc6 = xc6 + xc1
            # 转为rotation_dim向量，得到rot预测
            pred_rot = self.decrot(xc6)
        else:
            # 迭代优化法（direct_reg_rot=False）的核心思想是通过多次迭代逐步改进旋转预测，而不是一次性生成最终预测。
            # 这种方法更像是一个递归精炼过程，每次迭代都利用前一次的输出来改进结果。
            if self.rot_iterative_matmul:
                assert(self.rotation_dim == 6), self.rotation_dim
                for i in range(self.n_iter):
                    xc = torch.cat([xf, pred_rot],1).to(torch.float)
                    xc = self.fc_rot_1(xc)
                    xc = self.drop1(xc)
                    # if i < int(self.n_iter / 2):
                    #     skiplist2[i] = xc
                    # elif i == int(self.n_iter / 2) and self.n_iter % 2 != 0:
                    #     pass
                    # else:
                    #     xplus = skiplist2[i-int((self.n_iter+1) / 2)]
                    #     xc += xplus
                    xc = self.fc_rot_2(xc)
                    xc = self.drop2(xc)
                    pred_rot = rotmat_to_rot6d(rot6d_to_rotmat(self.decrot(xc)) @ rot6d_to_rotmat(pred_rot))
            else:
                for i in range(self.n_iter):
                    xc = torch.cat([xf, pred_rot],1).to(torch.float)
                    xc = self.fc_rot_1(xc)
                    xc = self.drop1(xc)
                    # if i < int(self.n_iter / 2):
                    #     skiplist2[i] = xc
                    # elif i == int(self.n_iter / 2) and self.n_iter % 2 != 0:
                    #     pass
                    # else:
                    #     xplus = skiplist2[i-int((self.n_iter+1) / 2)]
                    #     xc += xplus
                    xc = self.fc_rot_2(xc)
                    xc = self.drop2(xc)
                    pred_rot = self.decrot(xc) + pred_rot
        
        # 计算前向运动学
        if self.reference_keypoint_id == 0:
            pred_xyz_fk = self.robot.get_keypoints(pred_pose, pred_rot, pred_trans)
        else:
            pred_xyz_fk = self.robot.get_keypoints_root(pred_pose, pred_rot, pred_trans,root=self.reference_keypoint_id)
        
        if test_fps:
            # 同步当前 CUDA 流。这意味着它会等待所有在当前 CUDA 流中排队的操作完成，然后才继续执行后续的代码
            torch.cuda.current_stream().synchronize()
            t_end_other = time.time() 
            time_other = t_end_other - t_start_other
            time_whole = t_end_other - t_start_root
        
        if test_fps:
            return pred_pose, pred_rot, pred_trans, pred_root_uv, pred_depth, pred_uvd, pred_xyz_int, pred_xyz_fk, (time_root,time_other,time_whole)
        else:
            if self.multi_kp:
                # multi_kp多了个pred_depths，是DepthNet预测时预测了多个关键点的深度
                return pred_pose, pred_rot, pred_trans, pred_root_uv, pred_depth, pred_depths, pred_uvd, pred_xyz_int, pred_xyz_fk
            else:
                return pred_pose, pred_rot, pred_trans, pred_root_uv, pred_depth, pred_uvd, pred_xyz_int, pred_xyz_fk


    
def get_rootNetwithRegInt_model(init_params_dict, args, **kwargs):
    """ Constructs a rootNetwithRegInt model with ResNet/Hrnet backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if args.backbone_name not in ["resnet", "resnet50", "resnet34","resnet101", "hrnet", "hrnet32"]:
        raise(NotImplementedError)
    if args.rootnet_backbone_name not in ["resnet", "resnet50", "resnet34", "hrnet", "hrnet32"]:
        raise(NotImplementedError)
    
    model = RootNetwithRegInt(init_params_dict, args, **kwargs)
    
    model.reg_backbone.init_weights(args.backbone_name)
    if args.rootnet_backbone_name not in ["hrnet", "hrnet32"]:
        model.rootnet_backbone.init_weights(args.rootnet_backbone_name)
    
    # depthnet可以做预训练
    if args.pretrained_rootnet is not None:
        pretrained_path = args.pretrained_rootnet
        pretrained_checkpoint = torch.load(pretrained_path)
        print(f"Using {args.pretrained_rootnet} as pretrained rootnet weights for rootNetwithRegInt pipeline. ")
        pretrained_rootnet_weights = pretrained_checkpoint["model_state_dict"]
        pretrained_weights = {}
        for k, v in pretrained_rootnet_weights.items():
            if k.startswith("backbone"):
                new_k = k.replace("backbone", "rootnet_backbone")
            else:
                new_k = k
            pretrained_weights[new_k] = v
        # print(pretrained_weights.keys())
        model.load_state_dict(pretrained_weights, strict=False)
    else:
        print(f"Not using pretrained depthnet weights for the full network training stage. ")

    
    return model