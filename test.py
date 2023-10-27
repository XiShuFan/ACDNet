import os
import os.path
import argparse
import numpy as np
import torch
import PIL
from PIL import Image
import utils.save_image as save_img
from acdnet import ACDNet
import SimpleITK as sitk
import Li_process

parser = argparse.ArgumentParser(description="ACDNet_Test")
parser.add_argument("--model_dir", type=str, default="models/ACDNet_latest.pt", help='path to model file')
parser.add_argument('--N', type=int, default=6, help='the number of feature maps')
parser.add_argument('--Np', type=int, default=32, help='the number of channel concatenation')
parser.add_argument('--d', type=int, default=32, help='the number of convolutional filters in common dictionary D')
parser.add_argument('--num_res', type=int, default=3, help='Resblocks number in each ResNet')
parser.add_argument('--T', type=int, default=10, help='Stage number T')
parser.add_argument('--Mtau', type=float, default=1.5, help='for sparse feature map')
parser.add_argument('--etaM', type=float, default=1, help='stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5, help='stepsize for updating X')
parser.add_argument('--batchSize', type=int, default=1, help='testing input batch size')
opt = parser.parse_args()


def normalize(data):
    def normalized(X):
        maxX = np.max(X)
        minX = np.min(X)
        X = (X - minX) / (maxX - minX)
        return X

    def image_get_minmax():
        return 0.0, 1.0

    data = normalized(data)
    data_min, data_max = image_get_minmax()
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(data, 1)
    return data


def tohu(X):  # display window as [-175HU, 275HU]
    CT = (X - 0.192) * 1000 / 0.192
    CT_win = CT.clamp_(-175, 275)
    CT_winnorm = (CT_win + 175) / (275 + 175)
    return CT_winnorm


def test_image(img, mask):
    """
    slice_img: CT切片
    threshold: 金属伪影阈值
    """
    img_li = Li_process.Li_process(img, mask)
    img = normalize(img)
    img_li = normalize(img_li)
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, 1)
    non_mask = 1 - mask
    return torch.Tensor(img).cuda(), torch.Tensor(img_li).cuda(), torch.Tensor(non_mask).cuda()


def main():
    # Build model
    print('Loading model ...\n')
    model = ACDNet(opt).cuda()
    model.load_state_dict(torch.load(opt.model_dir))
    model.eval()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

    # TODO: 传入一个nii.gz文件，分片进行解析
    file_path = "source.nii.gz"
    vtk_img = sitk.ReadImage(file_path)
    array_img = sitk.GetArrayFromImage(vtk_img)

    # TODO: 统计强度值的范围，作证一个直方图，95%处认为是阈值
    threshold = np.percentile(array_img, 95)
    mask = array_img >= threshold

    # TODO: 预测
    Xma, XLI, M = test_image(array_img, mask)
    with torch.no_grad():
        torch.cuda.synchronize()
        ListX = []
        # 按照patch预测
        slices = np.arange(0, Xma.shape[0], 4)
        slices = np.append(slices, Xma.shape[0])[:2]
        for i in range(len(slices) - 1):
            start, end = slices[i], slices[i + 1]
            SliceX = model(Xma[start:end], XLI[start:end], M[start:end])
            ListX.append(SliceX)
            torch.cuda.empty_cache()
        ListX = torch.concat(ListX, dim=0)
    # assert ListX.shape[0] == Xma.shape[0], "维度不一致"
    Xoutclip = torch.clamp(ListX / 255.0, 0, 0.5)
    Xmaclip = torch.clamp(Xma / 255.0, 0, 0.5)
    Xoutnorm = Xoutclip / 0.5
    Xmanorm = Xmaclip / 0.5
    Xouthu = tohu(Xoutclip)
    Xmahu = tohu(Xmaclip)
    Xnorm = [Xoutnorm, Xmanorm]
    Xhu = [Xouthu, Xmahu]

    array_img_mar = Xoutnorm.cpu().numpy().squeeze()
    img_mar = sitk.GetImageFromArray(array_img_mar)
    # TODO: 输出到nii.gz中
    sitk.WriteImage(img_mar, "target.nii.gz")


if __name__ == "__main__":
    main()
