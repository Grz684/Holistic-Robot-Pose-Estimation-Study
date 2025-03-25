import torch
import numpy as np
import os

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    PerspectiveCameras, Textures
)

from os.path import exists
from roboticstoolbox.robot.ERobot import ERobot


class PandaArm():
    def __init__(self, urdf_file, dof=7): 
        self.robot = self.Panda(urdf_file)
        self.dof = dof
        
    def get_joint_RT(self, joint_angle):
        # 修改断言，根据指定的dof验证关节角度
        assert joint_angle.shape[0] == self.dof, f"Expected {self.dof} joint angles, got {joint_angle.shape[0]}"
        
        # 根据机器人类型确定链接索引列表
        if self.dof == 7:  # Panda
            link_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 9]
        elif self.dof == 6:  # Dofbot
            link_idx_list = [0, 1, 2, 3, 4, 5, 6, 7]  # 调整为实际的Dofbot链接索引
        elif self.dof == 4:  # OWI
            link_idx_list = [0, 1, 2, 3, 4]  # 调整为实际的OWI链接索引
        elif self.dof == 15:  # Baxter
            # 根据Baxter的链接结构定义
            link_idx_list = list(range(min(17, len(self.robot.links))))
        else:
            # 默认情况下假设链接索引与关节数量匹配
            link_idx_list = list(range(min(self.dof + 2, len(self.robot.links))))
        
        R_list = []
        t_list = []
        
        for i in range(len(link_idx_list)):
            link_idx = link_idx_list[i]
            T = self.robot.fkine(joint_angle, end=self.robot.links[link_idx], start=self.robot.links[0])
            R_list.append(T.R)
            t_list.append(T.t)

        return np.array(R_list), np.array(t_list)
        
    class Panda(ERobot):
        """
        Class that imports a URDF model
        """
        
        def __init__(self, urdf_file):

            links, name, urdf_string, urdf_filepath = self.URDF_read(urdf_file)

            super().__init__(
                links,
                name=name,
                manufacturer="Franka",
                urdf_string=urdf_string,
                urdf_filepath=urdf_filepath,
            )


class RobotMeshRenderer():
    """
    Class that render robot mesh with differentiable renderer
    """
    def __init__(self, focal_length, principal_point, image_size, robot, mesh_files, device):
        
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.device = device
        self.robot = robot
        self.mesh_files = mesh_files
        self.preload_verts = []
        self.preload_faces = []
        
        # 导入trimesh用于加载STL文件
        import trimesh

        # preload the mesh to save loading time
        for m_file in mesh_files:
            assert exists(m_file), f"File not found: {m_file}"
            
            # 获取文件扩展名
            file_extension = os.path.splitext(m_file)[1].lower()
            
            try:
                if file_extension == '.obj':
                    # 使用原有的OBJ加载方法
                    preload_verts_i, preload_faces_idx_i, _ = load_obj(m_file)
                    preload_faces_i = preload_faces_idx_i.verts_idx
                elif file_extension == '.stl':
                    # 使用trimesh加载STL文件
                    print(f"Loading STL file: {m_file}")
                    mesh = trimesh.load(m_file)
                    # 转换为PyTorch张量
                    preload_verts_i = torch.tensor(mesh.vertices, dtype=torch.float32)
                    # STL文件中的面是三角形，直接使用
                    preload_faces_i = torch.tensor(mesh.faces, dtype=torch.int64)
                else:
                    raise ValueError(f"Unsupported file format: {file_extension} for file {m_file}")
                
                self.preload_verts.append(preload_verts_i)
                self.preload_faces.append(preload_faces_i)
                print(f"Successfully loaded: {m_file}, vertices: {preload_verts_i.shape}, faces: {preload_faces_i.shape}")
                
            except Exception as e:
                print(f"Error loading {m_file}: {e}")
                # 如果加载失败，创建一个小的占位网格
                print("Creating placeholder mesh")
                preload_verts_i = torch.tensor([
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                    [0.0, 0.01, 0.0],
                    [0.0, 0.0, 0.01]
                ], dtype=torch.float32)
                preload_faces_i = torch.tensor([
                    [0, 1, 2],
                    [0, 2, 3],
                    [0, 3, 1],
                    [1, 3, 2]
                ], dtype=torch.int64)
                self.preload_verts.append(preload_verts_i)
                self.preload_faces.append(preload_faces_i)

        # 设置渲染器参数
        self.cameras = PerspectiveCameras(
                                     focal_length = [focal_length],
                                     principal_point = [principal_point],
                                     device=device, 
                                     in_ndc=False, image_size = [image_size]
                                     ) #  (height, width) !!!!!
        
        blend_params = BlendParams(sigma=1e-8, gamma=1e-8)
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
            max_faces_per_bin=100000,  # max_faces_per_bin=1000000,  
        )
        
        # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        
        
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000, 
        )
        # We can add a point light in front of the object. 
        lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=self.cameras, lights=lights)
        )
        
    def get_robot_mesh(self, joint_angle):
        
        R_list, t_list = self.robot.get_joint_RT(joint_angle)
        assert len(self.mesh_files) == R_list.shape[0] and len(self.mesh_files) == t_list.shape[0]

        verts_list = []
        faces_list = []
        verts_rgb_list = []
        verts_count = 0
        for i in range(len(self.mesh_files)):
            verts_i = self.preload_verts[i]
            faces_i = self.preload_faces[i]

            R = torch.tensor(R_list[i],dtype=torch.float32)
            t = torch.tensor(t_list[i],dtype=torch.float32)
            verts_i = verts_i @ R.T + t
            #verts_i = (R @ verts_i.T).T + t
            faces_i = faces_i + verts_count

            verts_count+=verts_i.shape[0]

            verts_list.append(verts_i.to(self.device))
            faces_list.append(faces_i.to(self.device))

            # Initialize each vertex to be white in color.
            color = torch.rand(3)
            verts_rgb_i = torch.ones_like(verts_i) * color  # (V, 3)
            verts_rgb_list.append(verts_rgb_i.to(self.device))



        verts = torch.concat(verts_list, dim=0)
        faces = torch.concat(faces_list, dim=0)

        verts_rgb = torch.concat(verts_rgb_list,dim=0)[None]
        textures = Textures(verts_rgb=verts_rgb)

        # Create a Meshes object
        robot_mesh = Meshes(
            verts=[verts.to(self.device)],   
            faces=[faces.to(self.device)], 
            textures=textures
        )
        
        return robot_mesh


    def get_robot_verts_and_faces(self, joint_angle):
        
        R_list, t_list = self.robot.get_joint_RT(joint_angle)
        assert len(self.mesh_files) == R_list.shape[0] and len(self.mesh_files) == t_list.shape[0]

        verts_list = []
        faces_list = []
        verts_rgb_list = []
        verts_count = 0
        for i in range(len(self.mesh_files)):
            verts_i = self.preload_verts[i]
            faces_i = self.preload_faces[i]

            R = torch.tensor(R_list[i],dtype=torch.float32)
            t = torch.tensor(t_list[i],dtype=torch.float32)
            verts_i = verts_i @ R.T + t
            #verts_i = (R @ verts_i.T).T + t
            faces_i = faces_i + verts_count

            verts_count+=verts_i.shape[0]

            verts_list.append(verts_i.to(self.device))
            faces_list.append(faces_i.to(self.device))

            # Initialize each vertex to be white in color.
            #color = torch.rand(3)
            #verts_rgb_i = torch.ones_like(verts_i) * color  # (V, 3)
            #verts_rgb_list.append(verts_rgb_i.to(self.device))

        verts = torch.concat(verts_list, dim=0)
        faces = torch.concat(faces_list, dim=0)

        
        return verts, faces