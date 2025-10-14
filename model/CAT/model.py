"""
该文件是CAT模型的主要定义
"""
import math
from sklearn.metrics import  average_precision_score

from tqdm import tqdm
from mindspore.ops import operations as P
import random
import numpy as np
from matplotlib.path import Path
import noise
import cv2
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from loss_fn import each_patch_loss_function
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal, Zero, One
from backbones import load_backbone_mindspore, load_backbone_pytorch
import timm.optim.optim_factory as optim_factory
from cat_utils import *
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
ms.set_context(mode=ms.PYNATIVE_MODE)
class SwinBlock(nn.Cell): # Swin-transformer 定义
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 6,
        window_size: int = 4,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        input_resolution: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = input_resolution * input_resolution + 1

        # 线性插值生成 drop_path 的 rate
        dpr = [x.item() for x in ops.linspace(Tensor(0, ms.float32), Tensor(drop_path_rate, ms.float32), depth)]
        
        # 使用标准的Transformer Block作为替代
        self.blocks = nn.CellList()
        for i in range(depth):
            block = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation="gelu",
                batch_first=True
            )
            self.blocks.append(block)

    def construct(self, x: Tensor):
        """
        x: Tensor of shape [B, 257, 768]
        return: Tensor of same shape
        """
        batch_size, num_tokens, channels = x.shape
        cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]  # [B,1,C], [B,256,C]
        
        # 对patch tokens应用transformer
        for block in self.blocks:
            patch_tokens = block(patch_tokens)
            
        return ops.cat([cls_token, patch_tokens], axis=1)  # [B,257,C]


class FPN(nn.Cell): # FPN层
    def __init__(self, scale_factors, decoder_embed_dim, fpn_output_dims, patch_size=None):
        super().__init__()
        assert len(fpn_output_dims) == len(scale_factors)
        self.scale_factors = scale_factors
        use_bias = False

        # 统一的内部融合通道
        self.fpn_dim = min(fpn_output_dims)
        
        # 1) 先构建各尺度的 C_i 分支
        c_layers, output_dims = [], []
        for scale in scale_factors:
            if scale == 4.0:
                layers = [
                    nn.Conv2dTranspose(decoder_embed_dim, decoder_embed_dim//2, 2, 2),
                    ConvLayerNorm(decoder_embed_dim//2), 
                    nn.GELU(),
                    nn.Conv2dTranspose(decoder_embed_dim//2, decoder_embed_dim//4, 2, 2),
                ]
                out_dim = decoder_embed_dim//4
            elif scale == 2.0:
                layers = [nn.Conv2dTranspose(decoder_embed_dim, decoder_embed_dim//2, 2, 2)]
                out_dim = decoder_embed_dim//2
            elif scale == 1.0:
                layers, out_dim = [], decoder_embed_dim
            elif scale == 0.5:
                layers, out_dim = [nn.MaxPool2d(2,2)], decoder_embed_dim
            else:
                raise NotImplementedError(f"scale={scale} not supported")
            c_layers.append(nn.SequentialCell(*layers))
            output_dims.append(out_dim)
        self.c_layers = nn.CellList(c_layers)

        # 2) lateral: 把每个 C_i 都映射到统一 fpn_dim
        self.lateral_convs = nn.CellList([
            Conv2d(output_dims[i], self.fpn_dim, kernel_size=1, bias=use_bias,
                   norm=ConvLayerNorm(self.fpn_dim))
            for i in range(len(scale_factors))
        ])
        
        # 3) 融合后的 3x3 conv
        self.fusion_convs = nn.CellList([
            Conv2d(self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1, bias=use_bias,
                   norm=ConvLayerNorm(self.fpn_dim))
            for _ in scale_factors
        ])

        # 4) 最终输出映射
        self.output_convs = nn.CellList([
            Conv2d(self.fpn_dim, fpn_output_dims[i], kernel_size=1, bias=use_bias,
                   norm=ConvLayerNorm(fpn_output_dims[i]))
            for i in range(len(scale_factors))
        ])

    def construct(self, x):
        # 1) 生成各尺度初始特征
        c_features = [layer(x) for layer in self.c_layers]

        # 2) lateral + 自底向上融合
        sorted_indices = sorted(range(len(self.scale_factors)),
                           key=lambda k: self.scale_factors[k])
        p_features = [None] * len(self.scale_factors)
        
        for i, idx in enumerate(sorted_indices):
            lateral = self.lateral_convs[idx](c_features[idx])
            if i == 0:
                p_features[idx] = lateral
            else:
                prev = p_features[sorted_indices[i-1]]
                scale_factor = self.scale_factors[idx] / self.scale_factors[sorted_indices[i-1]]
                # MindSpore的interpolate
                size = (int(prev.shape[-2] * scale_factor), int(prev.shape[-1] * scale_factor))
                upsampled = ops.interpolate(prev, size=size, mode='bilinear', align_corners=False)
                p_features[idx] = lateral + upsampled

        # 3) 融合卷积
        for idx in range(len(self.scale_factors)):
            p_features[idx] = self.fusion_convs[idx](p_features[idx])

        # 4) 最终映射到各自输出通道
        output_features = [
            self.output_convs[idx](p_features[idx])
            for idx in range(len(self.scale_factors))
        ]
        return output_features


class Conv2d(nn.Cell):# minspore版本的卷积层
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, group=1, bias=True, norm=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                            pad_mode='pad', padding=padding, dilation=dilation, 
                            group=group, has_bias=bias)
        self.norm = norm
        self.activation = activation

    def construct(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvLayerNorm(nn.Cell): # MindSpore版本的LayerNorm, 针对Conv2d输出
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.gamma = ms.Parameter(ops.ones(normalized_shape, ms.float32))
        self.beta = ms.Parameter(ops.zeros(normalized_shape, ms.float32))

    def construct(self, x):
        # x: [B, C, H, W]
        mean = x.mean(1, keep_dims=True)
        variance = (x - mean).pow(2).mean(1, keep_dims=True)
        x = (x - mean) / ops.sqrt(variance + self.eps)
        x = self.gamma[None, :, None, None] * x + self.beta[None, :, None, None]
        return x


class PatchEmbed(nn.Cell): # 简化的PatchEmbed实现
    def __init__(self, img_size=(224,224), patch_size=(16,16), in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size[0], 
                            stride=patch_size[0], has_bias=True, pad_mode='valid')
        # TODo: 目前仅仅正方形
    def construct(self, x):
        batch_size, height, width, channels = x.shape
        # 调整维度顺序: [B, H, W, C] -> [B, C, H, W]
        x = x.transpose(0, 3, 1, 2)
        x = self.proj(x)
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        x = x.flatten(2).transpose(0, 2, 1)
        return x


class TransformerBlock(nn.Cell): # TransformerBlock
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((embed_dim,))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, has_bias=qkv_bias)
        self.norm2 = norm_layer((embed_dim,))
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.SequentialCell([
            nn.Dense(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dense(hidden_dim, embed_dim)
        ])

    def construct(self, x):
        attended = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + attended
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """生成2D正弦余弦位置编码"""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """从网格生成2D正弦余弦位置编码"""
    assert embed_dim % 2 == 0
    
    # 使用一半维度用于height，一半用于width
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """从位置生成1D正弦余弦编码"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('i,j->ij', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class CATModel_mindspore(nn.Cell): # MindSpore版本的CAT模型
    def __init__(self, cfg, norm_layer=nn.LayerNorm):
        super().__init__()
        embed_dim = cfg.model_config.embed_dim
        patch_size = cfg.model_config.patch_size
        in_channels = cfg.in_channels
        num_heads = cfg.model_config.num_heads
        depth = cfg.model_config.depth
        mlp_ratio = cfg.model_config.mlp_ratio
        fpn_output_dims = cfg.model_config.FPN_output_dim
        scale_factors = cfg.model_config.scale_factors
        window_size=cfg.model_config.window_size
        qkv_bias=cfg.model_config.qkv_bias
        dpr=cfg.model_config.drop_path_rate
        input_res=cfg.model_config.input_resolution
        pretrain_image_size=cfg.model_config.pretrain_image_size
        self.swin_block = SwinBlock(embed_dim=embed_dim, depth=depth, num_heads=num_heads, window_size=window_size, 
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path_rate=dpr, 
                                    input_resolution=input_res)
        
        img_size = cfg.model_input_shape
        self.pretrain_num_patches = (pretrain_image_size[0] // patch_size[0]) * (pretrain_image_size[1] // patch_size[1])
        
        # MAE encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = ms.Parameter(ops.zeros((1, 1, embed_dim)))
        self.pos_embed = ms.Parameter(ops.zeros((1, self.pretrain_num_patches + 1, embed_dim)), 
                                    requires_grad=False)
        
        # Transformer blocks
        self.blocks = nn.CellList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer((embed_dim,))
        
        # Decoder
        decoder_embed_dim = embed_dim
        self.decoder_fpn_pos_embed = ms.Parameter(ops.zeros((1, num_patches + 1, decoder_embed_dim)),
                                                requires_grad=False)
        
        self.fpn = FPN(scale_factors=scale_factors, decoder_embed_dim=decoder_embed_dim, 
                      fpn_output_dims=fpn_output_dims, patch_size=patch_size)
        
        self.cfg = cfg
        self.initialize_weights()

    def initialize_weights(self):
        # 位置编码初始化
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                          int(self.pretrain_num_patches ** .5), cls_token=True)
        self.pos_embed.set_data(Tensor(pos_embed, ms.float32).unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_fpn_pos_embed.shape[-1],
                                                  int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_fpn_pos_embed.set_data(Tensor(decoder_pos_embed, ms.float32).unsqueeze(0))

        self.cls_token.set_data(initializer(Normal(sigma=0.02), self.cls_token.shape, self.cls_token.dtype))

    def generate_region(self, center_x, center_y, radius, scale, amplitude, num_points):
        """生成区域坐标"""
        thetas = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        r = radius + amplitude * np.array([noise.pnoise1(scale * t) for t in thetas])
        x = center_x + r * np.cos(thetas)
        y = center_y + r * np.sin(thetas)
        return x, y

    def custom_augment_preimage(self, img, probability=1):# 水滴增强算法
        img_np = img.asnumpy()
        if random.random() >= probability:
            return img_np
            
        height, width = img_np.shape[0], img_np.shape[1]
        center_x = random.uniform(0, height)
        center_y = random.uniform(0, width)
        radius = random.uniform(10, width//2)
        num_points = random.randint(25, 128)
        scale = random.uniform(radius//10, radius//2)
        amplitude = random.uniform(radius//10, radius//2)
        xs, ys = self.generate_region(center_x, center_y, radius, scale, amplitude, num_points)
        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)
        polygon = np.vstack((xs, ys)).T  
        poly_path = Path(polygon)
        max_attempts = random.randint(height*width//16, height*width)
        attempts = 0
        percent = random.uniform(0, 0.1)
        
        while attempts < max_attempts:
            rand_x = random.randint(0, height-1)
            rand_y = random.randint(0, width-1)
            if poly_path.contains_point((rand_x, rand_y)) and random.uniform(0, 1) < percent:
                img_np[rand_x][rand_y] = 1 
            attempts += 1
            
        return np.transpose(img_np, (1, 2, 0))

    def custom_augment(self, images):# 批量数据增强
        processed_images = [self.custom_augment_preimage(img.transpose(2, 0, 1)) for img in images] 
        processed_images = np.array(processed_images)
        return Tensor(processed_images, ms.float32)

    def forward_encoder(self, x):
        x = self.custom_augment(x)
        x = self.patch_embed(x)
        
        # 位置编码处理
        if self.patch_embed.num_patches != self.pretrain_num_patches:
            hw = (int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))
            pos_embed = self.pos_embed[:, 1:, :]
            new_size = (hw[0], hw[1])
            pos_embed_resized = ops.interpolate(
                pos_embed.reshape(1, int(math.sqrt(pos_embed.shape[1])), 
                                int(math.sqrt(pos_embed.shape[1])), -1).transpose(0, 3, 1, 2),
                size=new_size,
                mode='bilinear',
                align_corners=False
            ).transpose(0, 2, 3, 1).reshape(1, hw[0]*hw[1], -1)
            x = x + pos_embed_resized
        else:
            x = x + self.pos_embed[:, 1:, :]
            
        # 添加cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.tile((x.shape[0], 1, 1))
        x = ops.cat((cls_tokens, x), axis=1)
        
        # 通过Swin block和norm
        x = self.swin_block(x)
        x = self.norm(x)
        
        return x

    def forward_decoder_fpn(self, x):
        # 直接使用所有patch tokens（跳过cls token）
        x_patches = x[:, 1:, :]  # [B, N, C]
        
        # 添加位置编码
        x_patches = x_patches + self.decoder_fpn_pos_embed[:, 1:, :]

        # 序列转feature map
        batch_size, sequence_length, feature_dim = x_patches.shape
        height = width = int(sequence_length ** 0.5)
        feat_map = x_patches.transpose(0, 2, 1).reshape(batch_size, feature_dim, height, width)

        # FPN处理
        multi_scale_features = self.fpn(feat_map)

        # 映射为字典
        output = {
            layer_name: feature
            for layer_name, feature in zip(self.cfg.model_config.layers_to_extract_from,
                                      multi_scale_features)
        }
        return output

    def construct(self, images):
        if images.shape[1]<=3:
            images=ops.transpose(images, (0, 2, 3, 1))
        # print(f"Debug: {images.shape}")
        latent = self.forward_encoder(images)
        features = self.forward_decoder_fpn(latent)
        return features


def cat_base(cfg):
    model = CATModel_mindspore(cfg=cfg)
    return model
# 与模型名称同名的class是模型本体 　要求实现construct train _eval test 函数 手动调用load_config 初始化
def load_optimizer_mindspore(cfg, model):
    if cfg.optimizer == "adam":
        # 直接使用所有可训练参数
        optimizer = nn.Adam(
            params=model.trainable_params(),
            learning_rate=cfg.train_lr,
            beta1=getattr(cfg, 'adam_optimizer_beta1', 0.9),
            beta2=getattr(cfg, 'adam_optimizer_beta2', 0.95),
            weight_decay=cfg.train_weight_decay
        )
        return optimizer
    elif cfg.optimizer == "adamw":
        # 使用AdamWeightDecay（类似AdamW）
        optimizer = nn.AdamWeightDecay(
            params=model.trainable_params(),
            learning_rate=cfg.train_lr,
            beta1=getattr(cfg, 'adam_optimizer_beta1', 0.9),
            beta2=getattr(cfg, 'adam_optimizer_beta2', 0.95),
            weight_decay=cfg.train_weight_decay
        )
        return optimizer
    else:
        raise NotImplementedError(f"{cfg.optimizer} has not been supported yet!")
def cat_adjust_learning_rate(optimizer, epoch, cfg):# MindSpore版本的学习率调整
    if epoch < cfg.train_warmup_epoch:
        lr = cfg.train_lr * epoch / cfg.train_warmup_epoch
    else:
        # 保持原有的学习率调整逻辑
        if epoch<=80:
            lr=cfg.train_lr
        if epoch>80 and epoch<120:
            lr=cfg.train_lr/10
        if epoch>=120:
            lr=cfg.train_lr/100
        

    # MindSpore学习率更新方式
    if hasattr(optimizer, 'learning_rate'):
        # 对于nn.AdamWeightDecay等优化器
        optimizer.learning_rate = ms.Tensor(lr, ms.float32)
    elif hasattr(optimizer, 'get_lr') and hasattr(optimizer, 'set_lr'):
        # 对于其他优化器
        optimizer.set_lr(lr)
    
    print(f"Epoch {epoch}: learning rate = {lr}")
    return lr
def forward_hook_fn(cell_id, inputs, outputs, output_dict, layer_name):
    """
    MindSpore前向钩子函数
    Args:
        cell_id: 细胞单元信息
        inputs: 输入数据
        outputs: 输出数据
        output_dict: 存储输出的字典
        layer_name: 层名称
    """
    output_dict[layer_name] = outputs
    return None 
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import types



class MindSporeResNetHookManager:
    def __init__(self, model, layers_to_extract):
        self.model = model
        self.layers_to_extract = layers_to_extract
        self.feature_maps = {}
        self.hook_handles = []
        
    def create_hook_fn(self, layer_name):
        """为每个层创建独立的hook函数"""
        def hook_fn(cell_id, inputs, outputs):
            """钩子函数，捕获指定层的输出"""
            self.feature_maps[layer_name] = outputs
            return None
        return hook_fn
        
    def register_hooks(self):
        """注册前向钩子"""
        # 清空之前的钩子
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.feature_maps.clear()
        
        for layer_name in self.layers_to_extract:
            try:
                # 获取目标层
                target_cell = getattr(self.model, layer_name)
                
                # 为每个层创建独立的hook函数
                hook_func = self.create_hook_fn(layer_name)
                
                # 注册钩子
                handle = target_cell.register_forward_hook(hook_func)
                self.hook_handles.append(handle)
                print(f"Registered hook for layer: {layer_name}")
                
            except AttributeError as e:
                print(f"Error registering hook for {layer_name}: {e}")
                
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hook_handles:
            try:
                handle.remove()
            except:
                pass
        self.hook_handles = []
        
    def get_feature_maps(self):
        """获取捕获的特征图"""
        return self.feature_maps

    def clear_feature_maps(self):
        """清空特征图缓存"""
        self.feature_maps.clear()
class CAT_mindspore(nn.Cell):
    def __init__(self):
        super().__init__() 
        self.teacher_outputs_dict = {}
        self.hook_handles = []

    def load_config(self,cfg):
        self.cfg=cfg
        self.device=cfg.device
        self.cat_base=CATModel_mindspore(cfg)
        self.backbone=load_backbone_mindspore(cfg) 
        # freeze backbone with optimzer
        self._register_forward_hooks()
        if cfg.train_load_pretrained_model:
            ckpt=ms.load_checkpoint(cfg.train_pretrain_model_path)
            ms.load_param_into_net(self.cat_base, ckpt)
            print(f"Load pre-trained:{cfg.train_pretrain_model_path}")
        self.optimizer=load_optimizer_mindspore(cfg, self.cat_base)
    def _register_forward_hooks(self):
        """注册前向钩子"""
        if not self.backbone or not self.cfg:
            return
            
        # 初始化钩子管理器
        self.hook_manager = MindSporeResNetHookManager(
            self.backbone, 
            self.cfg.model_config.layers_to_extract_from
        )
        
        # 注册钩子
        self.hook_manager.register_hooks()
        
        # 为指定层注册钩子
    
    def train(self, train_dataloader):
        ms.set_context(device_target=self.cfg.device)

        
        # 1. 定义梯度计算函数
        grad_fn = ms.value_and_grad(self._forward_fn, None, self.optimizer.parameters)
        
        for cur_epoch in range(self.cfg.train_epochs):
            self.backbone.set_train(False)
            self.cat_base.set_train(True)
            cur_lr = cat_adjust_learning_rate(self.optimizer, cur_epoch, self.cfg)
            
            if (cur_epoch + 1) % 50 == 0:
                print(f"current lr is {cur_lr}")
                
            loss_list = []
            for batch_idx, image in enumerate(tqdm(train_dataloader)):
                if isinstance(image, dict):
                    image_tensor = image["image"]
                elif isinstance(image, list):
                    image_tensor=image[0]
                else:
                    image_tensor = image
                    
                if not isinstance(image_tensor, Tensor):
                    image_tensor = Tensor(image_tensor, ms.float32)
                    
                if len(image_tensor.shape) == 5:
                    image_tensor = ops.squeeze(image_tensor, axis=1)
                    
                # 清空上一轮的特征
                self.hook_manager.clear_feature_maps()
                
                # 教师网络前向传播
                _ = self.backbone(image_tensor)
                feature_maps = self.hook_manager.get_feature_maps()

                # 获取通过hook捕获的特征
                multi_scale_features = [
                    feature_maps[key] 
                    for key in self.cfg.model_config.layers_to_extract_from
                    if key in feature_maps
                ]
                if len(multi_scale_features) != len(self.cfg.model_config.layers_to_extract_from):
                    missing_layers = set(self.cfg.model_config.layers_to_extract_from) - set(feature_maps.keys())
                    print(f"Warning: Missing layers: {missing_layers}")
                    continue
                # print(f"debug2:{feature_maps.keys()}")
                # CAT模型前向传播并计算梯度
                loss, grads = grad_fn(image_tensor, multi_scale_features)
                
                # 使用梯度更新参数（MindSpore方式）
                self.optimizer(grads)
                
                # 记录损失
                loss_value = loss.asnumpy() if hasattr(loss, 'asnumpy') else float(loss)
                loss_list.append(loss_value)
            
            # 打印epoch统计信息
            if loss_list:
                avg_loss = np.mean(loss_list)
                print(f'Epoch [{cur_epoch + 1}/{self.cfg.train_epochs}], loss: {avg_loss:.4f}')
    
        return self.cat_base   
    def _save_model(self):
        pass
# 2. 添加前向计算函数
    def _forward_fn(self, image_tensor, multi_scale_features):
        # CAT模型前向传播
        reverse_features = self.cat_base(image_tensor)
        multi_scale_reverse_features = [
            reverse_features[key] 
            for key in self.cfg.model_config.layers_to_extract_from
        ]
        loss = each_patch_loss_function(multi_scale_features, multi_scale_reverse_features)
        return loss
    def _eval(self):
        pass 
    def test(self, test_dataloader):
        ms.set_context(device_target='CPU')
        self.backbone.set_train(False)
        self.cat_base.set_train(False)
        # Image-level
        labels_gt=[]
        labels_prediction=[]
        # Pixel-level
        masks_gt=[]
        masks_prediction=[]
        aupro_list=[]
        # Image-info
        img_path=[]
        img_name_list=[]
        for batch_idx, image in enumerate(tqdm(test_dataloader)):
            if isinstance(image, list):
                label_current = image[4].numpy()         # (batch_size,)
                mask_current = image[1].squeeze(1).numpy()     # (batch_size, H, W)
                labels_gt.extend(label_current.tolist())            # 扩展真实标签
                masks_gt.extend(mask_current)                       # 直接扩展 numpy 数组
                img_path.extend(image[2])
                img_name_list.extend(image[5])
                image_tensor=image[0]
                
                self.hook_manager.clear_feature_maps()
                if len(image_tensor.shape)==5:
                    image_tensor=ops.squeeze(image_tensor, axis=1)
                # print(f"Debug: {image_tensor.shape}")
                # 教师网络前向传播
                _ = self.backbone(image_tensor)
                feature_maps = self.hook_manager.get_feature_maps()

                # 获取通过hook捕获的特征
                multi_scale_features = [
                    feature_maps[key] 
                    for key in self.cfg.model_config.layers_to_extract_from
                    if key in feature_maps
                ]
                if len(multi_scale_features) != len(self.cfg.model_config.layers_to_extract_from):
                    missing_layers = set(self.cfg.model_config.layers_to_extract_from) - set(feature_maps.keys())
                    print(f"Warning: Missing layers: {missing_layers}")
                    continue
                reverse_features = self.cat_base(image_tensor)
                multi_scale_reverse_features = [
                    reverse_features[key] 
                    for key in self.cfg.model_config.layers_to_extract_from
                ]
                anomaly_map,_ =cal_anomaly_map(multi_scale_features, multi_scale_reverse_features, 
                                               image_tensor.shape[-1], amap_mode='a')
                anomaly_map=anomaly_map.numpy()
                for item in range(len(anomaly_map)):
                    anomaly_map[item] = gaussian_filter(anomaly_map[item], sigma=4)                
                labels_prediction.extend(np.max(anomaly_map.reshape(anomaly_map.shape[0], -1), 
                                            axis=1).tolist())
                masks_prediction.extend(anomaly_map)
                if set(mask_current.astype(int).flatten()) == {0, 1}:
                    aupro_list.extend(compute_pro(anomaly_map, mask_current.astype(int), label_current))
            else:
                raise Exception("Invalid test data format!")
             
        auroc_samples = round(roc_auc_score(labels_gt, labels_prediction), 3)
        print(f"Debug RE: {len(masks_gt)} {masks_gt[0].shape} {len(masks_prediction)} {masks_prediction[0].shape}")
        pixel_scores = compute_pixelwise_retrieval_metrics(masks_prediction, masks_gt)
        auroc_pixel = pixel_scores["auroc"]
        mean_aupro = round(np.mean(aupro_list), 3)

        AP_det = average_precision_score(labels_gt, labels_prediction)
        masks_gt_flat = np.concatenate([mask.flatten() for mask in masks_gt]).astype(np.int32)          # 展平并拼接
        masks_pred_flat = np.concatenate([pred.flatten() for pred in masks_prediction]) # 展平并拼接
        AP_loc = average_precision_score(masks_gt_flat, masks_pred_flat)

        # print(f"Debug :I_AUROC:{auroc_samples} P_AUROC:{auroc_pixel} PRO:{mean_aupro} AP_det:{AP_det} AP_loc:{AP_loc}")
        return {
            "I_AUROC":auroc_samples,
            "P_AUROC":auroc_pixel,
            "PRO":mean_aupro,
            "AP_det":AP_det,
            "AP_loc":AP_loc
        }
    def construct(self,images):
        pass

