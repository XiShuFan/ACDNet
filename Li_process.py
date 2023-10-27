import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import os

def linear_interpolation(img, iter=10, mask=None):
    """使用线性插值修复图像"""
    img_temp = np.copy(img)

    # 如果提供了mask，则先对mask区域进行重建
    if mask is not None:
        img_temp[mask == 1] = 0

    # 使用一个2D卷积核来获取边界像素的平均值
    kernel = np.ones((1, 3, 3))
    kernel[0, 1, 1] = 0

    #iter 根据需要设置迭代次数
    for _ in range(iter):
        sum_neigh = ndimage.convolve(img_temp, kernel, mode='nearest')
        num_neigh = ndimage.convolve(np.ones_like(img_temp), kernel, mode='nearest')

        valid_idx = num_neigh != 0
        corrected_values = np.divide(sum_neigh, num_neigh, out=np.zeros_like(img, dtype=float), where=valid_idx)

        if mask is not None:
            img_temp[mask == 1] = np.where(valid_idx & (mask == 1), corrected_values, img_temp)[mask == 1]
        else:
            img_temp = np.where(valid_idx, corrected_values, img_temp)

    return img_temp


def Li_process(img, mask):
    # 首先对mask部分进行重建
    img_corrected = linear_interpolation(img, 10, mask)
    # 然后对整个重建后的图像进行一次线性插值处理
    img_corrected_whole = linear_interpolation(img_corrected, 10)

    return img_corrected_whole
