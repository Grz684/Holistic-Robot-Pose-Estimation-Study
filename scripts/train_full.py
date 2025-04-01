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
import torch.distributed as dist


def get_local_rank():
    """Get LOCAL_RANK environment variable"""
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return 0


def train_full(args):
    # Get local rank for distributed training
    args.local_rank = get_local_rank()
    print(f"Using local_rank: {args.local_rank}")
    
    torch.autograd.set_detect_anomaly(True)
    set_random_seed(808)
    
    # Add random seed + rank to ensure different processes use different seeds
    if args.distributed:
        set_random_seed(808 + args.local_rank)
    
    save_folder, ckpt_folder, log_folder, writer = create_logger(args)
    
    urdf_robot_name = args.urdf_robot_name
    robot = URDFRobot(urdf_robot_name)
 
    device_id = args.device_id
    
    # Initialize distributed environment
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Get dataloaders with distributed sampler if distributed
    ds_iter_train, test_loader_dict = get_dataloaders(args)
    
    init_param_dict = {
        "robot_type": urdf_robot_name,
        "pose_params": INITIAL_JOINT_ANGLE,
        "cam_params": np.eye(4, dtype=float),
        "init_pose_from_mean": True
    }

    # 是否共享backbone
    if args.use_rootnet_with_reg_int_shared_backbone:
        print("regression and integral shared backbone, with rootnet 2 backbones in total")
        model = get_rootNetwithRegInt_model(init_param_dict, args)
    else:
        assert 0
    
    # Move model to device before initializing optimizer
    model.to(device)
    model.float()
    
    # If using DDP, wrap the model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    curr_max_auc = 0.0
    curr_max_auc_4real = {"azure": 0.0, "kinect": 0.0, "realsense": 0.0, "orb": 0.0}
    start_epoch, last_epoch, end_epoch = 0, -1, args.n_epochs
    
    # Resume training if needed
    if args.resume_run:
        start_epoch, last_epoch, curr_max_auc, curr_max_auc_4real = resume_run(args, model, optimizer, device)
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(args, optimizer, last_epoch)
 
    for epoch in range(start_epoch, end_epoch + 1):
        # Only print on main process
        if not args.distributed or dist.get_rank() == 0:
            print(f'In epoch {epoch + 1}, script: full network training (JointNet/RotationNet/KeypoinNet/DepthNet)')
        
        # Set epoch for distributed sampler
        if args.distributed:
            ds_iter_train.sampler.set_epoch(epoch)
            # Add barrier to ensure all processes are synchronized before starting the epoch
            dist.barrier()
        
        model.train()
        
        # Only show progress bar on main process
        if not args.distributed or dist.get_rank() == 0:
            iterator = tqdm(ds_iter_train, dynamic_ncols=True)
        else:
            iterator = ds_iter_train
            
        # Initialize meters for tracking losses
        losses = AverageValueMeter()
        losses_pose, losses_rot, losses_trans, losses_uv, losses_depth, losses_error2d, losses_error3d, losses_error2d_int, losses_error3d_int, losses_error3d_align = \
            AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter(), AverageValueMeter()
        
        for batchid, sample in enumerate(iterator):
            optimizer.zero_grad()
            loss, loss_dict = farward_loss(args=args, input_batch=sample, model=model, robot=robot, device=device, device_id=[args.local_rank] if args.distributed else device_id, train=True)
            loss.backward()
            if args.clip_gradient is not None:
                clipping_value = args.clip_gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            
            # Track losses
            losses.add(loss.detach().cpu().numpy())
            losses_pose.add(loss_dict["loss_joint"].detach().cpu().numpy())
            losses_rot.add(loss_dict["loss_rot"].detach().cpu().numpy())
            losses_trans.add(loss_dict["loss_trans"].detach().cpu().numpy())
            losses_uv.add(loss_dict["loss_uv"].detach().cpu().numpy())
            losses_depth.add(loss_dict["loss_depth"].detach().cpu().numpy())
            losses_error2d.add(loss_dict["loss_error2d"].detach().cpu().numpy())
            losses_error3d.add(loss_dict["loss_error3d"].detach().cpu().numpy())
            losses_error2d_int.add(loss_dict["loss_error2d_int"].detach().cpu().numpy())
            losses_error3d_int.add(loss_dict["loss_error3d_int"].detach().cpu().numpy())
            losses_error3d_align.add(loss_dict["loss_error3d_align"].detach().cpu().numpy())

            # Log metrics only on main process
            # 训练指标只记录主进程作为参考，不进行汇总，不影响模型训练
            if (batchid+1) % 100 == 0 and (not args.distributed or dist.get_rank() == 0):
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
                
                # Reset meters
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
                
            # Log learning rates
            if not args.distributed or dist.get_rank() == 0:
                writer.add_scalar('LR/learning_rate_opti', optimizer.param_groups[0]['lr'], epoch * len(ds_iter_train) + batchid + 1)
                if len(optimizer.param_groups) > 1:
                    for pgid in range(1, len(optimizer.param_groups)):
                        writer.add_scalar(f'LR/learning_rate_opti_{pgid}', optimizer.param_groups[pgid]['lr'], epoch * len(ds_iter_train) + batchid + 1)
        
        if args.use_schedule:
            lr_scheduler.step()
            
        # Validation - all processes participate, but results gathered on main process
        auc_adds = {}
        
        # Sync before validation
        if args.distributed:
            dist.barrier()
            
        for dsname, loader in test_loader_dict.items():
            if args.distributed:
                # Distributed validation
                # validate函数会汇总所有进程的结果，不过只有主进程会输出汇总结果，以给模型保存提供参考
                auc_add = validate(args=args, epoch=epoch, dsname=dsname, loader=loader, model=model,
                                   robot=robot, writer=writer, device=device, 
                                   device_id=[args.local_rank])
            else:
                auc_add = validate(args=args, epoch=epoch, dsname=dsname, loader=loader, model=model,
                                  robot=robot, writer=writer, device=device, device_id=device_id)
            auc_adds[dsname] = auc_add
        
        # Save checkpoint only on main process
        if not args.distributed or dist.get_rank() == 0:
            save_checkpoint(args=args, auc_adds=auc_adds,
                            model=model, optimizer=optimizer,
                            ckpt_folder=ckpt_folder,
                            epoch=epoch, lr_scheduler=lr_scheduler,
                            curr_max_auc=curr_max_auc,
                            curr_max_auc_4real=curr_max_auc_4real)
        
        # Sync after checkpoint saving
        if args.distributed:
            dist.barrier()
                  
    # Final message only on main process
    if not args.distributed or dist.get_rank() == 0:
        print("Training Finished!")
        writer.flush()
