"""
CAT模型需要用的工具模块
"""
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor

import numpy as np
from numpy import ndarray
from sklearn import metrics
import cv2
import os
import pandas as pd
from skimage import measure
from statistics import mean
from sklearn.metrics import auc
import random
import logging
import math

from PIL import Image

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    """
    计算异常图
    
    Args:
        fs_list: 学生特征列表 [Tensor, Tensor, ...]
        ft_list: 教师特征列表 [Tensor, Tensor, ...]  
        out_size: 输出特征图大小
        amap_mode: 融合模式 'mul' 或 'add'
    
    Returns:
        anomaly_map: 融合后的异常图 [B, H, W]
        a_map_list: 各层异常图列表
    """
    # 获取batch size
    batch_size = fs_list[0].shape[0]
    
    # 初始化异常图
    if amap_mode == 'mul':
        anomaly_map = ops.ones((batch_size, out_size, out_size), ms.float32)
    else:
        anomaly_map = ops.zeros((batch_size, out_size, out_size), ms.float32)
    
    a_map_list = []
    
    for i in range(len(ft_list)):
        fs = fs_list[i]  # 学生特征 [B, C, H, W]
        ft = ft_list[i]  # 教师特征 [B, C, H, W]
        
        # 计算余弦相似度
        # 1. 归一化特征
        fs_norm = ops.L2Normalize(axis=1)(fs)  # 沿通道维度归一化
        ft_norm = ops.L2Normalize(axis=1)(ft)
        
        # 2. 计算点积得到余弦相似度
        cosine_sim = ops.reduce_sum(fs_norm * ft_norm, axis=1)  # [B, H, W]
        
        # 3. 计算异常得分 (1 - 余弦相似度)
        a_map = 1 - cosine_sim  # [B, H, W]
        
        # 4. 增加通道维度并插值到目标大小
        a_map = ops.expand_dims(a_map, 1)  # [B, 1, H, W]
        
        # 双线性插值
        # resize_bilinear =  nn.ResizeBilinear((out_size, out_size))
        a_map =  ops.interpolate(a_map, (out_size,out_size), mode="bilinear")  # [B, 1, out_size, out_size]
        
        a_map = ops.squeeze(a_map, 1)  # [B, out_size, out_size]
        
        a_map_list.append(a_map)
        
        # 融合异常图
        if amap_mode == 'mul':
            anomaly_map = anomaly_map * a_map
        else:
            anomaly_map = anomaly_map + a_map
            
    return anomaly_map, a_map_list

def compute_pro(anomaly_map: ndarray, gt_mask: ndarray, label: ndarray, num_th: int = 200): # 计算PRO
    assert isinstance(anomaly_map, ndarray), "type(amaps) must be ndarray"
    assert isinstance(gt_mask, ndarray), "type(masks) must be ndarray"
    assert anomaly_map.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert gt_mask.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert anomaly_map.shape == gt_mask.shape, "amaps.shape and masks.shape must be same"
    assert set(gt_mask.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    current_amap = anomaly_map[label != 0]
    current_mask = gt_mask[label != 0].astype(int)

    binary_amaps = np.zeros_like(current_amap[0], dtype=np.bool_)
    pro_auc_list = []

    for anomaly_mask, mask in zip(current_amap, current_mask):
        df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
        min_th = anomaly_mask.min()
        max_th = anomaly_mask.max()
        delta = (max_th - min_th) / num_th

        for th in np.arange(min_th, max_th, delta):
            binary_amaps[anomaly_mask <= th] = 0
            binary_amaps[anomaly_mask > th] = 1

            pros = []
            # for connect region
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amaps[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

            inverse_masks = 1 - mask
            fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()

            fpr = fp_pixels / inverse_masks.sum()

            df = df._append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

        # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
        df = df[df["fpr"] < 0.3]
        df["fpr"] = df["fpr"] / df["fpr"].max()

        pro_auc = auc(df["fpr"], df["pro"])

        pro_auc_list.append(pro_auc)

    return pro_auc_list

def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    统一计算相关测试参数
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    # flatten
    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    mean_AP = metrics.average_precision_score(flat_ground_truth_masks.astype(int),
                                              flat_anomaly_segmentations)

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "mean_AP": mean_AP
    }