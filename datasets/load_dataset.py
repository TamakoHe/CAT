import sys
import os
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _is_image_file(path) -> bool:
    return path.suffix.lower() in IMG_EXTS
def _get_dataloaders_mindposre(cfg, mode):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from dataloader_mindspore import get_dataloaders
    return get_dataloaders(cfg,mode)

def _get_dataloaders_pytorch(cfg, mode):
    pass 
def get_dataloaders(cfg, mode='train'): # 加载mindspore版本的数据加载器
    if cfg.model_framework=="mindspore": 
        return _get_dataloaders_mindposre(cfg, mode)
    elif cfg.model_framework=="pytorch":
        pass 
    else:
        raise NotImplementedError(f"Framwork {cfg.model_framework} has not been supported yet!")
