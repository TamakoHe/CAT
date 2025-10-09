# dataset_mvtec_flat.py
import imp
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Inter
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import os

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


class CATDataset:
    """
    Dataset for flat layout:
    Dataset/
        train/           # *.png (train images only)
        test/            # *.png (test images)
        ground_truth/    # *.png masks, filenames match test/ filenames exactly

    mode:
      - "train": returns image tensor (for self-supervised training)
      - "test" : returns (image tensor, mask tensor)

    Args:
      root: dataset root dir (str or Path)
      mode: "train" or "test"
      transform: callable applied to PIL image -> tensor (default: ToTensor())
      mask_transform: callable applied to PIL mask -> tensor (default: binary LongTensor 0/1)
      return_paths: if True, returns dict with paths for debugging/visualization
      strict: if True, raise FileNotFoundError when a mask for a test image is missing;
              if False, return zero mask and print a warning (default: True)
      mask_dtype: mindspore.dtype for returned mask (ms.int32 or ms.float32)
    """
    def __init__(
        self,
        root: Union[str, Path],
        mode: str = "train",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        return_paths: bool = False,
        strict: bool = True,
        mask_dtype: ms.Type = ms.int32,
    ):
        assert mode in ("train", "test"), "mode must be 'train' or 'test'"
        self.root = Path(root)
        self.mode = mode
        self.return_paths = return_paths
        self.strict = strict
        self.mask_dtype = mask_dtype

        # default transforms
        self.transform = transform
        # default mask transform: convert grayscale PIL -> binary 0/1 tensor with requested dtype
        if mask_transform is not None:
            self.mask_transform = mask_transform
        else:
            def default_mask_transform(pil: Image.Image):
                arr = np.asarray(pil.convert("L"))
                bin_arr = (arr > 0).astype(np.uint8)  # any non-zero pixel considered defect
                if mask_dtype == ms.float32:
                    return bin_arr.astype(np.float32)
                else:
                    return bin_arr.astype(np.int32)
            self.mask_transform = default_mask_transform

        # check directories
        self.train_dir = self.root / "train"
        self.test_dir = self.root / "test"
        self.gt_dir = self.root / "ground_truth"

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        if self.mode == "train":
            if not self.train_dir.exists():
                raise FileNotFoundError(f"train/ not found under {self.root}")
            self.images = sorted([p for p in self.train_dir.iterdir() if p.is_file() and _is_image_file(p)])
            if len(self.images) == 0:
                raise RuntimeError(f"No images found in {self.train_dir}")
        else:
            # test mode
            if not self.test_dir.exists():
                raise FileNotFoundError(f"test/ not found under {self.root}")
            if not self.gt_dir.exists() and strict:
                raise FileNotFoundError(f"ground_truth/ not found under {self.root}")
            self.test_images = sorted([p for p in self.test_dir.iterdir() if p.is_file() and _is_image_file(p)])
            if len(self.test_images) == 0:
                raise RuntimeError(f"No images found in {self.test_dir}")
            # build mapping for gt files (flat)
            self.gt_map = {p.name: p for p in sorted(self.gt_dir.iterdir()) if p.is_file() and _is_image_file(p)} if self.gt_dir.exists() else {}
            # prepare pairs list
            self.pairs = []
            warned = False
            for img in self.test_images:
                mask = self.gt_map.get(img.name)
                if mask is None:
                    if self.strict:
                        raise FileNotFoundError(f"Mask for test image '{img.name}' not found in ground_truth/")
                    else:
                        if not warned:
                            print(f"[FlatMVTecLikeDataset] Warning: some masks missing; zero masks will be returned. (First missing: {img.name})")
                            warned = True
                self.pairs.append((img, mask))

    def __len__(self):
        return len(self.images) if self.mode == "train" else len(self.pairs)

    def __getitem__(self, idx: int):
        if self.mode == "train":
            img_path = self.images[idx]
            img = Image.open(img_path).convert("RGB")
            
            if self.transform:
                img_t = self.transform(img)
            else:
                img_t = np.array(img).astype(np.float32) / 255.0
                img_t = np.transpose(img_t, (2, 0, 1))  # HWC to CHW
                
            if self.return_paths:
                return img_t, str(img_path)
            return img_t
        else:
            img_path, mask_path = self.pairs[idx]
            img = Image.open(img_path).convert("RGB")
            
            if self.transform:
                img_t = self.transform(img)
            else:
                img_t = np.array(img).astype(np.float32) / 255.0
                img_t = np.transpose(img_t, (2, 0, 1))  # HWC to CHW
            
            is_anomaly = False
            if mask_path is not None and mask_path.exists():
                mask_pil = Image.open(mask_path).convert('L')
                mask_t = self.mask_transform(mask_pil)
                is_anomaly = int(np.max(mask_t) != 0)
            else:
                # create zero mask same HxW as image
                w, h = img.size
                if self.mask_dtype == ms.float32:
                    mask_t = np.zeros((h, w), dtype=np.float32)
                else:
                    mask_t = np.zeros((h, w), dtype=np.int32)
                    
            if self.return_paths:
                return img_t, mask_t, str(img_path), (str(mask_path) if mask_path is not None else ""), is_anomaly, "/".join(str(img_path).split("/")[-4:])
            return img_t, mask_t

def _dataloader(cfg, dataset, mode):
    """创建 MindSpore 数据加载器"""
    if mode == "train":
        # 训练模式数据加载器
        dataloader = ds.GeneratorDataset(
            dataset,
            column_names=["image"] if not dataset.return_paths else ["data", "path"],
            shuffle=True,
            num_parallel_workers=cfg.train_num_workers
        )
    else:
        # 测试模式数据加载器
        if dataset.return_paths:
            column_names = ["image", "mask", "image_path", "mask_path", "is_anomaly", "image_name"]
        else:
            column_names = ["image", "mask"]
            
        dataloader = ds.GeneratorDataset(
            dataset,
            column_names=column_names,
            shuffle=False,
            num_parallel_workers=cfg.test_num_workers
        )
    
    batch_size = cfg.train_batch_size if mode == "train" else cfg.test_batch_size
    dataloader = dataloader.batch(batch_size, drop_remainder=(mode == "train"))
    
    return dataloader


def get_dataloaders(cfg, mode='train'):
    if mode == 'train':
        # 训练数据转换
        trans = transforms.Compose([
            vision.RandomResizedCrop(cfg.model_input_shape, 
                                   scale=(cfg.model_config.DA_low_limit, cfg.model_config.DA_up_limit),
                                   interpolation=Inter.BICUBIC),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, is_hwc=False)
        ])
        print(cfg.dataset_path_train)
        dataset = CATDataset(root=cfg.dataset_path_train, mode="train", transform=trans, return_paths=False)
        return _dataloader(cfg, dataset, mode)
    
    elif mode == 'test':
        # 测试数据转换
        trans = transforms.Compose([
            vision.Resize(cfg.model_input_shape,Inter.BICUBIC),
            vision.ToTensor(),
            vision.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, is_hwc=False)
        ])
        
        size = cfg.model_input_shape
        mask_transform = transforms.Compose([
            vision.Resize(size,Inter.BICUBIC),
            vision.CenterCrop(size),
            lambda x: np.array(x).astype(np.int32)  # 简单的mask转换
        ])

        dataset = CATDataset(
            root=cfg.dataset_path_test, 
            mode="test", 
            transform=trans,
            mask_transform=mask_transform,
            return_paths=True, 
            strict=False, 
            mask_dtype=ms.int32
        )
        
        print("test samples:", len(dataset))
        return _dataloader(cfg, dataset, mode)
    
    else:
        raise NotImplementedError("Unknown mode")
# TODO: 要加上val 的 loader哦