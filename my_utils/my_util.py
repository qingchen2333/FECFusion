import datetime
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch import nn


#################################################################################
# 图像处理函数：ImageProcessing
# 对融合初图像进行一些处理
#################################################################################
def ImageProcessing(fusion_image):
    fusion_image = torch.clamp(fusion_image, max=1.0, min=0.0)      # 限制在[0,1]，不然会出现耀斑
    # 转回CPU，因为之前可能加载到GPU了
    fused_image = fusion_image.cpu().detach().numpy()
    fused_image = fused_image.transpose((0, 2, 3, 1))
    # 均衡像素强度
    # fused_image = (fused_image - np.min(fused_image)) / (np.max(fused_image) - np.min(fused_image))
    fused_image = np.uint8(255.0 * fused_image)
    return fused_image


#################################################################################
# 运行时间处理函数：RunningTime
# 对训练的运行时间进行一些处理
#################################################################################
class RunningTime:
    def __init__(self):
        self.start_time = time.time()
        self.init_time = time.time()
        self.end_time = 0
        self.this_time = 0
        self.total_time = 0
        self.now_it = 0
        self.eta = 0

    def runtime(self, this_epo, it, dataset_size, epoch):
        self.end_time = time.time()
        self.this_time = self.end_time - self.start_time
        self.total_time = self.end_time - self.init_time
        self.now_it = dataset_size * this_epo + it + 1
        self.eta = int((dataset_size * epoch - self.now_it) * (self.total_time / self.now_it))
        self.eta = str(datetime.timedelta(seconds=self.eta))
        self.start_time = self.end_time
        return self.eta, self.this_time, self.now_it


#################################################################################
# 单个特征可视化函数
# 在模型的forward阶段返回一个特征值，然后将其保存为各通道一起的灰度图像
#################################################################################
def feature_save(fmap, save_path=None, fmap_size=None, i=None):
    if fmap_size is None:
        _, _, H, W = fmap.shape
        fmap_size = [H, W]
    fmap = torch.unsqueeze(fmap[-1], 0)  # 选取倒数第一张图，保持原来的尺寸，只要一张图的tensor
    fmap.transpose_(0, 1)  # 等价于 x = x.transpose(0,1), 把B和C的维度调换
    nrow = int(np.sqrt(fmap.shape[0]))
    fmap = F.interpolate(fmap, size=fmap_size, mode="bilinear")  # 改变数组尺寸，变成[64, 64]
    fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
    if not os.path.exists(save_path):     # 如果没有路径文件夹，就创建文件夹
        os.makedirs(save_path)
        print('Making fused_path {}'.format(save_path))
    save_path = os.path.join(save_path, 'feature{}.png'.format(i))
    vutils.save_image(fmap_grid, save_path)
    print('Save feature map {}'.format(save_path))


#################################################################################
# 构建K=7，sigmaS=25，sigmac=7.5的双边滤波
#################################################################################
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)  # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))  # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()  # 归一化
    return kernel


def bilateralFilter(batch_img, ksize, sigmaColor=0., sigmaSpace=0.):
    device = batch_img.device
    if sigmaSpace == 0:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor == 0:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
    # patches.shape: B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6

    # 求出像素亮度差
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)
    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
    # 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix


# -----------------------------------------------#
# 构建K=7，主要调用这个
# -----------------------------------------------#
class bilateral_Filter(nn.Module):
    def __init__(self, ksize=5, sigmacolor=0., sigmaspace=0.):
        super(bilateral_Filter, self).__init__()
        self.ksize = ksize
        self.sigmacolor = sigmacolor
        self.sigmaspace = sigmaspace

    def forward(self, x):
        x = bilateralFilter(x, self.ksize, self.sigmacolor, self.sigmaspace)
        return x


#################################################################################
# 色域转换
#   底层函数：RGB2YCrCb、YCrCb2RGB
#   顶层函数：ColorSpaceTransform
#################################################################################
# 底层函数，色域转换基本函数
def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(input_im.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(input_im.device)
    temp = (im_flat + bias).mm(mat).to(input_im.device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    return out


# 顶层函数，model选择转换模式，images_input输入图像
def ColorSpaceTransform(model, images_input):
    if model == 'RGB2YCrCb':
        images_vis_ycrcb = RGB2YCrCb(images_input)
        return images_vis_ycrcb
    elif model == 'YCrCb2RGB':
        fusion_image = YCrCb2RGB(images_input)
        return fusion_image


def algorithm_runtime(runtime):
    t_avg = np.mean(runtime)
    t_std = np.std(runtime)
    print('t_avg is {:5f}±{:5f}s'.format(t_avg, t_std))


if __name__ == '__main__':
    x = torch.tensor(np.random.rand(8, 64, 60, 60).astype(np.float32))
    feature_save(x, save_path='../feature_show')
