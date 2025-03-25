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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', '-c', type=str, required=True, default='configs/cfg.yaml', help="hyperparameters path")
    args = parser.parse_args()
    cfg = make_cfg(args)
    
    print("-------------------   config for this experiment   -------------------")
    print(cfg)
    print("----------------------------------------------------------------------")

    # 论文里的说法：我们首先在合成训练数据集上对DepthNet进行了100个历元的预训练，采用1e-4的学习率，
    # 并完全依赖于深度的地面真实监督。随后，我们基于预训练的DepthNet对整个模型进行了额外的100个epoch的训练，
    # 学习率为1e-4，衰减率为0.95。Adam优化器在每个阶段都用于优化网络参数，动量设置为0.9。
    # 由于深度会影响所有关键点的全局偏移，因此预训练的DepthNet可以作为训练其他网络的合理起点。
    # 在Panda真实数据集上，我们进一步对真实世界的图像进行自我监督训练，以1e-6的学习率克服sim到真实域的差距。
    # 这种微调过程只使用图像数据，不使用地面实况标签。
    
    if cfg.use_rootnet_with_reg_int_shared_backbone:
        print(f"\n pipeline: full network training (JointNet/RotationNet/KeypoinNet/DepthNet) \n")
        train_full(cfg)
    
    elif cfg.use_rootnet:
        print("\n pipeline: training DepthNet only \n")
        train_depthnet(cfg)
        
    elif cfg.use_sim2real:
        print("\n pipeline: self-supervised training on real datasets \n")
        train_sim2real(cfg)
    
    elif cfg.use_sim2real_real:
        print("\n pipeline: self-supervised training on my real datasets \n")
        # train_sim2real_real(cfg)
        
        
    
