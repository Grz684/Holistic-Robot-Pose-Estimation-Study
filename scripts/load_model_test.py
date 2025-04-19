import argparse
import sys
import os
import torch
from collections import OrderedDict
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 只让程序看到第4个GPU

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# path
save_folder = "models/"
model_path = os.path.join(save_folder, f"curr_best_root_depth_model.pk")

# make models and set initialization 
ckpt = torch.load(model_path) 

state_dict = ckpt['model_state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # Remove "module." prefix if it exists
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v
print("This model was saved from epoch:", ckpt["epoch"])
print(ckpt["loss"])