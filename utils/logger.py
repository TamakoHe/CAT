import mindspore as ms 
from datetime import datetime
import torch
import os
import re
import toml
"""
这个文件主要的目的是保存训练和测试的日志
"""
def get_next_run_id(top_dir: str) -> int:
    """
    获取 top_dir 下所有形如 runX 的子目录，并返回下一个编号 N+1。
    如果不存在任何 run 目录，则返回 1。
    """
    max_id = 0
    pattern = re.compile(r'^run(\d+)$')

    for name in os.listdir(top_dir):
        match = pattern.match(name)
        if match:
            run_id = int(match.group(1))
            max_id = max(max_id, run_id)

    return max_id + 1
def save_results(cfg, model, ckpt_name, info:dict):
    top_dir=cfg.output_dir
    os.makedirs(top_dir, exist_ok=True)
    nid=get_next_run_id(top_dir)
    os.makedirs(os.path.join(top_dir, f"./run{nid}", "weight"),exist_ok=True)
    save_path=os.path.join(top_dir, f"./run{nid}", "weight", ckpt_name)  # 创建文件夹
    if cfg.model_framework=="mindspore":
        ms.save_checkpoint(model,save_path) # 保存权重(mindspore)
    elif cfg.model_framework=="pytorch":
        torch.save(model.state_dict(), save_path)
    else:
        raise NotImplementedError(f"{cfg.model_framework} has not supported yet!")
    now = datetime.now() 
    time_str = now.strftime("%Y%m%d_%H%M%S") 
    info["save_time"]=time_str
    with open(os.path.join(top_dir, f"run{nid}", "run_info.toml"), "w") as out:
        toml.dump(info, out) # 保存测试结果
    