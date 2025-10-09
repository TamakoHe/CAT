import sys
import os
import mindspore.dataset as ds
import numpy as np
# 添加上层目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import config 
from datasets import load_dataset
if __name__=="__main__":
    debug_toml_path="/root/cat-mindspore/config/config.toml"
    cfg=config.Config()
    cfg.load_toml(toml_path=debug_toml_path)
    ld_train=load_dataset.get_dataloaders(cfg, "train")
    ld_test=load_dataset.get_dataloaders(cfg,"test")
    for data in ld_train.create_dict_iterator(output_numpy=True):
        print(data['image'].shape)
        break
    for data in ld_test.create_dict_iterator(output_numpy=True):
        print(data.keys())
        break
    