import sys
import os
import mindspore.dataset as ds
import numpy as np
import mindspore as ms
import mindspore.ops as ops
# 添加上层目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import config 
from datasets import load_dataset
from model.CAT import model
if __name__=="__main__":
    debug_toml_path="/root/autodl-tmp/cat-mindspore/config/config.toml"
    cfg=config.Config()
    cfg.load_toml(toml_path=debug_toml_path)
    debug_cm=model.CAT_mindspore()
    debug_cm.load_config(cfg)
    debug_ld=load_dataset.get_dataloaders(cfg)
    debug_cm.train(debug_ld)
    
    
