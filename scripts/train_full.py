import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from lib.core.function import farward_loss, validate
from lib.dataset.const import INITIAL_JOINT_ANGLE
from lib.models.full_net import get_rootNetwithRegInt_model
from lib.utils.urdf_robot import URDFRobot
from lib.utils.utils import set_random_seed, create_logger, get_dataloaders, get_scheduler, resume_run, save_checkpoint
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

# 导入Accelerate库
from accelerate import Accelerator
from accelerate.utils import set_seed

def train_full(args):
    # 初始化Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16',  # 可以设置为'fp16'或'bf16'启用混合精度训练
    )
    
    # 设置随机种子（使用Accelerate的函数）
    set_seed(808)
    
    # 只在主进程创建logger
    if accelerator.is_main_process:
        save_folder, ckpt_folder, log_folder, writer = create_logger(args)
    else:
        writer = None
    
    urdf_robot_name = args.urdf_robot_name
    robot = URDFRobot(urdf_robot_name)
 
    # 使用accelerator的device，不需要手动设置device
    device = accelerator.device
    device_id = getattr(args, 'device_id', 0)  # 保留device_id用于与其他函数兼容

    # 获取数据加载器
    # 需要修改get_dataloaders函数，见下面的说明
    ds_iter_train, test_loader_dict = get_dataloaders(args)
    
    init_param_dict = {
        "robot_type" : urdf_robot_name,
        "pose_params": INITIAL_JOINT_ANGLE,
        "cam_params": np.eye(4,dtype=float),
        "init_pose_from_mean": True
    }
    
    if args.use_rootnet_with_reg_int_shared_backbone:
        accelerator.print("regression and integral shared backbone, with rootnet 2 backbones in total")
        model = get_rootNetwithRegInt_model(init_param_dict, args)
    else:
        assert 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    curr_max_auc = 0.0
    curr_max_auc_4real = { "azure": 0.0, "kinect": 0.0, "realsense": 0.0, "orb": 0.0 }
    start_epoch, last_epoch, end_epoch = 0, -1, args.n_epochs
    
    if args.resume_run:
        # 注意：resume_run函数可能需要小修改以适应device变化
        start_epoch, last_epoch, curr_max_auc, curr_max_auc_4real = resume_run(args, model, optimizer, device)
        
    lr_scheduler = get_scheduler(args, optimizer, last_epoch)
    
    # 使用Accelerate准备模型、优化器和数据加载器
    model, optimizer, ds_iter_train, lr_scheduler = accelerator.prepare(
        model, optimizer, ds_iter_train, lr_scheduler
    )
    
    # 训练循环
    for epoch in range(start_epoch, end_epoch + 1):
        accelerator.print(f'In epoch {epoch + 1}, script: full network training (JointNet/RotationNet/KeypoinNet/DepthNet)')
        
        model.train()
        # 使用tqdm创建进度条（只在主进程显示）
        iterator = tqdm(ds_iter_train, dynamic_ncols=True, disable=not accelerator.is_local_main_process)
        
        losses = AverageValueMeter()
        losses_pose, losses_rot, losses_trans, losses_uv, losses_depth = \
            AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter()
        losses_error2d, losses_error3d, losses_error2d_int, losses_error3d_int, losses_error3d_align = \
            AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter()
        
        
        for batchid, sample in enumerate(iterator):
                
            # 前向传播和计算损失
            loss, loss_dict = farward_loss(args=args, input_batch=sample, model=model, 
                                          robot=robot, device=device, device_id=device_id, train=True)
            
            # 使用Accelerate进行反向传播
            accelerator.backward(loss)
            
            # 梯度裁剪
            if args.clip_gradient is not None:
                clipping_value = args.clip_gradient
                accelerator.clip_grad_norm_(model.parameters(), clipping_value)
                
            # 参数更新
            optimizer.step()
            optimizer.zero_grad()
            
            # 收集所有进程的损失值
            gathered_loss = accelerator.gather(loss.detach()).mean().item()
            losses.add(gathered_loss)
            
            # 收集各项损失
            for loss_name, meter in [
                ("loss_joint", losses_pose),
                ("loss_rot", losses_rot),
                ("loss_trans", losses_trans),
                ("loss_uv", losses_uv),
                ("loss_depth", losses_depth),
                ("loss_error2d", losses_error2d),
                ("loss_error3d", losses_error3d),
                ("loss_error2d_int", losses_error2d_int),
                ("loss_error3d_int", losses_error3d_int),
                ("loss_error3d_align", losses_error3d_align)
            ]:
                gathered_item = accelerator.gather(loss_dict[loss_name].detach()).mean().item()
                meter.add(gathered_item)
            
            # 更新TensorBoard（只在主进程）
            if accelerator.is_main_process and (batchid+1) % 100 == 0:
                writer.add_scalar('Train/loss', losses.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/pose_loss', losses_pose.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/rot_loss', losses_rot.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/trans_loss', losses_trans.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/uv_loss', losses_uv.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/depth_loss', losses_depth.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error2d_loss', losses_error2d.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error3d_loss', losses_error3d.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error2d_int_loss', losses_error2d_int.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error3d_int_loss', losses_error3d_int.mean, epoch * len(ds_iter_train) + batchid + 1)
                writer.add_scalar('Train/error3d_align_loss', losses_error3d_align.mean, epoch * len(ds_iter_train) + batchid + 1)
                
                # 记录学习率
                writer.add_scalar('LR/learning_rate_opti', optimizer.param_groups[0]['lr'], epoch * len(ds_iter_train) + batchid + 1)
                if len(optimizer.param_groups) > 1:
                    for pgid in range(1,len(optimizer.param_groups)):
                        writer.add_scalar(f'LR/learning_rate_opti_{pgid}', optimizer.param_groups[pgid]['lr'], epoch * len(ds_iter_train) + batchid + 1)
                
                # 重置meters
                losses.reset()
                losses_pose.reset()
                losses_rot.reset()
                losses_trans.reset()
                losses_uv.reset()
                losses_depth.reset()
                losses_error2d.reset()
                losses_error3d.reset()
                losses_error2d_int.reset()
                losses_error3d_int.reset()
                losses_error3d_align.reset()
        
        # 更新学习率调度器
        if args.use_schedule:
            lr_scheduler.step()
        
        # 等待所有进程完成训练步骤
        accelerator.wait_for_everyone()
            
        # 验证和保存模型（只在主进程）
        if accelerator.is_main_process:
            
            # 获取未包装的模型用于验证
            unwrapped_model = accelerator.unwrap_model(model)
            # 获取模型的实际设备
            model_device = next(unwrapped_model.parameters()).device
            
            # 运行验证
            auc_adds = {}
            for dsname, loader in test_loader_dict.items():
                auc_add = validate(args=args, epoch=epoch, dsname=dsname, loader=loader, 
                                  model=unwrapped_model, robot=robot, writer=writer, 
                                  device=model_device, device_id=device_id)
                auc_adds[dsname] = auc_add

            # 保存检查点
            save_checkpoint(args=args, auc_adds=auc_adds, 
                           model=unwrapped_model, optimizer=optimizer, 
                           ckpt_folder=ckpt_folder, 
                           epoch=epoch, lr_scheduler=lr_scheduler, 
                           curr_max_auc=curr_max_auc, 
                           curr_max_auc_4real=curr_max_auc_4real)
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        accelerator.print("Training Finished !")
        if writer:
            writer.flush()