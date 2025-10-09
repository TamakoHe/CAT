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
    debug_toml_path="/root/cat-mindspore/config/config.toml"
    cfg=config.Config()
    cfg.load_toml(toml_path=debug_toml_path)
    debug_model=model.cat_base(cfg)
    debug_input=ops.randn((16,256,256,3))
    debug_output=debug_model(debug_input)
    print(debug_output)