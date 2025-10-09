# safe_cell_del.py — 在程序最先被 import
import mindspore.nn.cell as _ms_cell

_orig_cell_del = getattr(_ms_cell.Cell, "__del__", None)

def _safe_cell_del(self):
    try:
        if _orig_cell_del:
            _orig_cell_del(self)
    except Exception:
        # 忽略析构时的异常（避免 "NoneType has no attribute 'total_instance_count'"）
        # 可在这里记录日志：print("Ignored Cell.__del__ exception")
        return

# 覆盖原 __del__
_ms_cell.Cell.__del__ = _safe_cell_del

from mindcv.models.model_factory import create_model
from mindcv.models import registry

# 你原来的 PyTorch/timm 对应名 → MindCV 支持的 model_name 映射（你可以扩充这个表）
_MINDCV_MAPPING = {
    "resnet50": "resnet50",
    "resnet18": "resnet18",
    "resnet101": "resnet101",
    "resnetv2_50_bit": "resnetv2_50x3_bitm",
    "resnetv2_50_21k": "resnetv2_50x3_bitm_in21k",
    "efficientnet_b3": "efficientnet_b3",  # MindCV 支持 “efficientnet” 族模型 :contentReference[oaicite:2]{index=2}
    "vgg19": "vgg19",
    "vgg19_bn": "vgg19",  # MindCV 的 vgg 接口里默认无 batchnorm 或带 batchnorm 版本可选 :contentReference[oaicite:3]{index=3}
    # … 你可以继续为你的 _BACKBONES 列表做映射
}

def load_ms_via_mindcv(name: str, pretrained: bool = True, num_classes: int = 1000, in_channels: int = 3, checkpoint_path: str = ""):
    """
    用 MindCV 加载模型。如果 mapping 不支持则抛错或 fallback。
    name: 你的 backbone 名称（如 "resnet50", "vit_base"…）
    pretrained: 是否加载预训练权重（等价于 MindCV 的 pretrained=True）
    num_classes: 分类数量
    in_channels: 输入通道数
    checkpoint_path: 如果有本地 ckpt 文件可以指定（与 pretrained 互斥）
    """
    if name not in _MINDCV_MAPPING:
        raise ValueError(f"Backbone {name} is not supported via MindCV mapping.")
    model_name = _MINDCV_MAPPING[name]

    # 创建模型：MindCV 的 create_model 支持 pretrained 和 checkpoint_path 两种方式
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        checkpoint_path=checkpoint_path
    )
    return model


def load_backbone_mindspore(cfg):
    return load_ms_via_mindcv(cfg.model_config.backbone, True, 5, in_channels=cfg.in_channels)
def load_backbone_pytorch():
    pass
if __name__ == "__main__":
    # 加载模型
    m = load_ms_via_mindcv("resnet50")

    # ===== 验证推理部分 =====
    import mindspore as ms
    import numpy as np

    # 设置推理模式（防止进入训练图）
    m.set_train(True)

    # 构造一个随机输入张量，符合模型输入维度 [batch, channel, height, width]
    dummy_input = ms.Tensor(np.random.randn(1, 3, 224, 224), ms.float32)

    # 前向推理
    output = m(dummy_input)

    # 输出结果维度与部分数值
    print("✅ 推理成功！输出 shape:", output.shape)
    print("前 5 个 logits:", output[0, :5])