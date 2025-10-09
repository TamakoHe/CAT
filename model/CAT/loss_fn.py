import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

def each_patch_loss_function(a, b):
    """
    修正版的多尺度特征余弦相似度损失函数
    """
    total_loss = 0
    # print(f"Debug: la:{len(a)} lb:{len(b)}")
    
    for i in range(len(a)):
        # 调整维度: [B, C, H, W] -> [B, H, W, C]
        a_tem = a[i].transpose(0, 2, 3, 1)
        b_tem = b[i].transpose(0, 2, 3, 1)
        
        # 重塑为 [N, C] 其中 N = B * H * W
        a_flat = a_tem.reshape(-1, a_tem.shape[-1])
        b_flat = b_tem.reshape(-1, b_tem.shape[-1])
        
        # 使用ops.cosine_similarity直接计算:cite[1]
        similarity = ops.cosine_similarity(a_flat, b_flat, dim=1)  # 在特征维度计算相似度
        sim_loss = ops.reduce_mean(1.0 - similarity)
        
        total_loss += sim_loss
    
    # 返回平均损失
    return total_loss / len(a)