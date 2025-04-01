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
    config_path = 'configs/dofbot/full.yaml'
    parser.set_defaults(config=config_path)
    args = parser.parse_args()
    cfg = make_cfg(args)
    
    print("-------------------   config for this experiment   -------------------")
    print(cfg)
    print("----------------------------------------------------------------------")

    # train_full(cfg)
    train_full(cfg)