import argparse
import importlib
from utils import config,logger
from datasets import load_dataset
import random
import numpy as np
import mindspore as ms
import torch
def rng_all(seed):
    random.seed(seed)     
    np.random.seed(seed)  # NumPy 随机库
    ms.set_seed(seed)     # MindSpore 全局随机种子
def get_parser():
    parser = argparse.ArgumentParser(description="This is our beloved model")
    parser.add_argument("--config", type=str,required=True, help="main config file path")
    return parser
if __name__ == "__main__":
    args = get_parser().parse_args() # 获取CLI的参数(只有一个统一配置文件的路径)
    config_path=args.config 
    cfg=config.Config() 
    cfg.load_toml(config_path) # 加载toml配置文件
    rng_all(cfg.rng_seed) # 全局设置随机种子 
    module = importlib.import_module(f"model.{cfg.model_name}.model") 
    model_class_ms = getattr(module, f"{cfg.model_name}_mindspore") # 动态导入配置指定的模型
    if cfg.model_framework=="mindspore": # 只实现了mindspore的版本
        our_model=model_class_ms() 
        our_model.load_config(cfg) # 加载模型配置
        if cfg.enable_train:
            our_dataloader=load_dataset.get_dataloaders(cfg) #加载训练用的数据加载器
            trained_model=our_model.train(our_dataloader)  # 进行训练
        if cfg.enable_test:
            our_dataloader_test=load_dataset.get_dataloaders(cfg, "test") # 加载测试用的数据加载器
            res=our_model.test(our_dataloader_test) # 测试结果
        if cfg.train_save_ckpt and cfg.enable_train:
                logger.save_results(cfg,trained_model, "last.mindpt", {
                    
                } if not cfg.enable_test else res) # 保存训练的权重和测试结果
                