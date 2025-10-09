import mindspore.nn as nn
from mindcv.models import resnet50
from mindspore import Tensor
import mindspore.ops as ops
import mindspore as ms 

class FeatureExtractorResNet50(nn.Cell):
    def __init__(self, ckpt_path=None):
        super().__init__()
        # 加载预训练的ResNet50
        self.backbone = resnet50()
        if ckpt_path:
            ms.load_checkpoint(ckpt_path, self.backbone)
        
        # 设置不需要分类头
        self.backbone.fc = nn.Identity()  # 或者直接移除分类头
        
        # 提取中间层所需的子模块
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        
    def construct(self, x):
        # 初始卷积和池化
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.max_pool(x)
        
        # 获取各层输出
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        
        # 返回中间层结果
        return {
            'layer1': layer1_out,  # 形状: (B, 256, H/4, W/4)
            'layer2': layer2_out,  # 形状: (B, 512, H/8, W/8)  
            'layer3': layer3_out   # 形状: (B, 1024, H/16, W/16)
        }

# 使用示例
model = FeatureExtractorResNet50("")
model.set_train(True)  # 设置为评估模式

# 假设输入是形状为 (1, 3, 224, 224) 的Tensor
input_tensor = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
features = model(input_tensor)

print("Layer1 output shape:", features['layer1'].shape)
print("Layer2 output shape:", features['layer2'].shape)
print("Layer3 output shape:", features['layer3'].shape)
