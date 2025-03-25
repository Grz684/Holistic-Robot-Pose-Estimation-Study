import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.transforms import uvd_to_xyz


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))

def norm_heatmap_hrnet(norm_type, heatmap, tau=5, sample_num=1):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == 'sampling':
        heatmap = heatmap.reshape(*shape[:2], -1)

        eps = torch.rand_like(heatmap)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau

        gumbel_heatmap = F.softmax(gumbel_heatmap, 2)
        return gumbel_heatmap.reshape(*shape)
    elif norm_type == 'multiple_sampling':

        heatmap = heatmap.reshape(*shape[:2], 1, -1)

        eps = torch.rand(*heatmap.shape[:2], sample_num, heatmap.shape[3], device=heatmap.device)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau
        gumbel_heatmap = F.softmax(gumbel_heatmap, 3)
        gumbel_heatmap = gumbel_heatmap.reshape(shape[0], shape[1], sample_num, shape[2])

        # [B, S, K, -1]
        return gumbel_heatmap.transpose(1, 2)
    else:
        raise NotImplementedError
    
def norm_heatmap_resnet(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError
    
def get_intrinsic_matrix_batch(f, c, bsz, inv=False):
        
        intrinsic_matrix = torch.zeros((bsz, 3, 3)).to(torch.float)

        if inv:
            intrinsic_matrix[:, 0, 0] = 1.0 / f[0].to(float)
            intrinsic_matrix[:, 0, 2] = - c[0].to(float) / f[0].to(float)
            intrinsic_matrix[:, 1, 1] = 1.0 / f[1].to(float)
            intrinsic_matrix[:, 1, 2] = - c[1].to(float) / f[1].to(float)
            intrinsic_matrix[:, 2, 2] = 1
        else:
            intrinsic_matrix[:, 0, 0] = f[0]
            intrinsic_matrix[:, 0, 2] = c[0]
            intrinsic_matrix[:, 1, 1] = f[1]
            intrinsic_matrix[:, 1, 2] = c[1]
            intrinsic_matrix[:, 2, 2] = 1

        return intrinsic_matrix.cuda(device=0)
    
class HeatmapIntegralPose(nn.Module):
    """
    This module takes in heatmap output and performs soft-argmax(integral operation).
    """
    # 模块的主要作用是接收热图输出并执行软最大值（积分操作），从而估计关节的三维坐标。
    # 具体来说，它将热图转换为关节的 UVD（深度图）坐标，
    # 然后将这些 UVD 坐标转换为 XYZ（世界坐标系）坐标。
    def __init__(self, backbone, **kwargs):
        super(HeatmapIntegralPose, self).__init__()
        self.backbone_name = backbone
        self.norm_type = kwargs["norm_type"]
        self.num_joints = kwargs["num_joints"]
        # UVD 坐标的维度
        self.depth_dim = kwargs["depth_dim"]
        self.height_dim = kwargs["height_dim"]
        self.width_dim = kwargs["width_dim"]
        
        self.rootid = kwargs["rootid"] if "rootid" in kwargs else 0
        self.fixroot = kwargs["fixroot"] if "fixroot" in kwargs else False
        
        # self.focal_length = kwargs['FOCAL_LENGTH'] if 'FOCAL_LENGTH' in kwargs else 320
        bbox_3d_shape = kwargs['bbox_3d_shape'] if 'bbox_3d_shape' in kwargs else (2300, 2300, 2300)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 1e-3
        self.image_size = kwargs["image_size"]
    
    
    def forward(self, out, flip_test=False, **kwargs):
        """
        Adapted from https://github.com/Jeff-sjtu/HybrIK/tree/main/hybrik/models
        """
        
        K = kwargs["K"]
        root_trans = kwargs["root_trans"]
        batch_size = out.shape[0]
        inv_k = get_intrinsic_matrix_batch((K[:,0,0],K[:,1,1]), (K[:,0,2],K[:,1,2]), bsz=batch_size, inv=True)
        
        if self.backbone_name in ["resnet", "resnet34", "resnet50"]:
            # out = out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
            # 重塑输出张量为(批次大小, 关节数, 其他维度合并)
            # 这种"展平-softmax-加权平均"的范式已成为3D关节点定位的标准做法
            out = out.reshape((out.shape[0], self.num_joints, -1))
            out = norm_heatmap_resnet(self.norm_type, out)
            assert out.dim() == 3, out.shape
            # out已经通过softmax归一化，其沿dim=2的和应该已经是1，再次除以沿dim=2的和实际上是不必要的操作
            heatmaps = out / out.sum(dim=2, keepdim=True)
            # 张量维度转化为B×N×D×H′×W′
            heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))
            # 这行代码对深度维度(2)和高度维度(3)求和，结果是在x轴方向(宽度方向)的边缘概率分布
            hm_x0 = heatmaps.sum((2, 3)) # (B, K, W)
            hm_y0 = heatmaps.sum((2, 4)) # (B, K, H)
            hm_z0 = heatmaps.sum((3, 4)) # (B, K, D)

            # 如果image_size为256，depth_dim为64，height_dim为64，width_dim为64
            # 如果W, H, 和D尺寸相同，代码会意外地工作，但概念上仍然是错误的
            range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device)
            
            hm_x = hm_x0 * range_tensor
            hm_y = hm_y0 * range_tensor
            hm_z = hm_z0 * range_tensor

            # 这种方法是对完整3D软极大值操作的近似，但大大提高了效率：
            # 完整3D体积需要O(DHW)的计算复杂度，分解方法只需要O(D+H+W)的复杂度
            coord_x = hm_x.sum(dim=2, keepdim=True)
            coord_y = hm_y.sum(dim=2, keepdim=True)
            coord_z = hm_z.sum(dim=2, keepdim=True)
            
            # 除以维度大小：将坐标从像素/体素单位转换为相对坐标（0到1之间）
            # 减去0.5：将坐标系原点移到图像/体积中心，生成范围在-0.5到0.5之间的标准化坐标
            coord_x = coord_x / float(self.width_dim) - 0.5
            coord_y = coord_y / float(self.height_dim) - 0.5
            coord_z = coord_z / float(self.depth_dim) - 0.5

            #  -0.5 ~ 0.5
            pred_uvd_jts = torch.cat((coord_x, coord_y, coord_z), dim=2)  
            if self.fixroot: 
                pred_uvd_jts[:,self.rootid,2] = 0.0

            # 形状从[batch_size, num_joints, 3]变为[batch_size, num_joints*3]
            pred_uvd_jts_flat = pred_uvd_jts.reshape(batch_size, -1)
            
            # return_relative=False，返回绝对坐标
            pred_xyz_jts = uvd_to_xyz(uvd_jts=pred_uvd_jts, image_size=self.image_size, intrinsic_matrix_inverse=inv_k, 
                                           root_trans=root_trans, depth_factor=self.depth_factor, return_relative=False)
            
            # pred_uvd_jts_back = xyz_to_uvd(xyz_jts=pred_xyz_jts, image_size=self.image_size, intrinsic_matrix=K, 
            #                                root_trans=root_trans, depth_factor=self.depth_factor, return_relative=False)
            # print("(pred_uvd_jts-pred_uvd_jts_back).sum()",(pred_uvd_jts.cuda()-pred_uvd_jts_back.cuda()).sum())
            
            return pred_uvd_jts, pred_xyz_jts
        
        elif self.backbone_name == "hrnet" or self.backbone_name == "hrnet32" or self.backbone_name == "hrnet48":
            out = out.reshape((out.shape[0], self.num_joints, -1))
            heatmaps = norm_heatmap_hrnet(self.norm_type, out)
            assert heatmaps.dim() == 3, heatmaps.shape
            heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))

            hm_x0 = heatmaps.sum((2, 3))  # (B, K, W)
            hm_y0 = heatmaps.sum((2, 4))  # (B, K, H)
            hm_z0 = heatmaps.sum((3, 4))  # (B, K, D)

            range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device).unsqueeze(-1)
            # hm_x = hm_x0 * range_tensor
            # hm_y = hm_y0 * range_tensor
            # hm_z = hm_z0 * range_tensor

            # coord_x = hm_x.sum(dim=2, keepdim=True)
            # coord_y = hm_y.sum(dim=2, keepdim=True)
            # coord_z = hm_z.sum(dim=2, keepdim=True)
            coord_x = hm_x0.matmul(range_tensor)
            coord_y = hm_y0.matmul(range_tensor)
            coord_z = hm_z0.matmul(range_tensor)

            coord_x = coord_x / float(self.width_dim) - 0.5
            coord_y = coord_y / float(self.height_dim) - 0.5
            coord_z = coord_z / float(self.depth_dim) - 0.5

            #  -0.5 ~ 0.5
            pred_uvd_jts = torch.cat((coord_x, coord_y, coord_z), dim=2)
            if self.fixroot: 
                pred_uvd_jts[:,self.rootid,2] = 0.0
            pred_uvd_jts_flat = pred_uvd_jts.reshape(batch_size, -1)
            
            pred_xyz_jts = uvd_to_xyz(uvd_jts=pred_uvd_jts, image_size=self.image_size, intrinsic_matrix_inverse=inv_k, 
                                           root_trans=root_trans, depth_factor=self.depth_factor, return_relative=False)
            
            # pred_uvd_jts_back = xyz_to_uvd(xyz_jts=pred_xyz_jts, image_size=self.image_size, intrinsic_matrix=K, 
            #                                root_trans=root_trans, depth_factor=self.depth_factor, return_relative=False)
            # print("(pred_uvd_jts-pred_uvd_jts_back).sum()",(pred_uvd_jts.cuda()-pred_uvd_jts_back.cuda()).sum())
            
            return pred_uvd_jts, pred_xyz_jts
        
        else:
            raise(NotImplementedError)
        
        
class HeatmapIntegralJoint(nn.Module):
    """
    This module takes in heatmap output and performs soft-argmax(integral operation).
    """
    def __init__(self, backbone, **kwargs):
        super(HeatmapIntegralJoint, self).__init__()
        self.backbone_name = backbone
        self.norm_type = kwargs["norm_type"]
        self.dof = kwargs["dof"]
        self.joint_bounds = kwargs["joint_bounds"]
        assert self.joint_bounds.shape == (self.dof, 2), self.joint_bounds.shape
    
    
    def forward(self, out, **kwargs):
        """
        Adapted from https://github.com/Jeff-sjtu/HybrIK/tree/main/hybrik/models
        """
        
        batch_size = out.shape[0]
        
        if self.backbone_name in ["resnet34", "resnet50"]:
            out = out.reshape(batch_size, self.dof, -1)
            out = norm_heatmap_resnet(self.norm_type, out)
            assert out.dim() == 3, out.shape
            heatmaps = out / out.sum(dim=2, keepdim=True)
            heatmaps = heatmaps.reshape((heatmaps.shape[0], self.dof, -1)) # no depth dimension
            
            resolution = heatmaps.shape[-1]
            range_tensor = torch.arange(resolution, dtype=torch.float32, device=heatmaps.device).reshape(1,1,resolution)
            hm_int = heatmaps * range_tensor
            # 求和得到坐标期望值（在resolution坐标轴上）
            coord_joint_raw = hm_int.sum(dim=2, keepdim=True)
            coord_joint = coord_joint_raw / float(resolution) # 0~1
            
            bounds = self.joint_bounds.reshape(1,self.dof,2).cuda()
            jointrange = bounds[:,:,[1]] - bounds[:,:,[0]]

            # 乘以关节范围，加上关节下限
            joints = coord_joint * jointrange + bounds[:,:,[0]]
            
            return joints.squeeze(-1)
        
        else:
            raise(NotImplementedError)
    