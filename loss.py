# -*- encoding: utf-8 -*-
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_ssim import ssim
from my_utils.my_util import bilateral_Filter
from scipy.signal import convolve2d


# sobel边缘算子
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()  # 不可训练，为了保持网络完整性
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.Reflect = nn.ReflectionPad2d(1)

    def forward(self, x):
        x = self.Reflect(x)
        sobelx = F.conv2d(x, self.weightx, padding=0)
        sobely = F.conv2d(x, self.weighty, padding=0)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        # A = torch.mean(self.sobelconv(image_A))
        # B = torch.mean(self.sobelconv(image_B))
        # weight_A = A * A / (A * A + B * B)
        # weight_B = 1.0 - weight_A
        # #  计算权重，让纹理（梯度）更多的图有更高的权重
        # Loss_SSIM = (weight_A * (1 - ssim(image_A, image_fused)) + weight_B * (1 - ssim(image_B, image_fused))) * 0.5

        Loss_SSIM = (1 - ssim(image_A, image_fused)) / 2 + (1 - ssim(image_B, image_fused)) / 2
        return Loss_SSIM


class Fusionloss(nn.Module):
    def __init__(self, weight=None):
        super(Fusionloss, self).__init__()
        if weight is None:
            weight = [10, 35, 1, 10]
        self.sobelconv = Sobelxy()
        self.bila = bilateral_Filter(ksize=11, sigmacolor=0.05, sigmaspace=8.0)
        self.L_SSIM = L_SSIM()
        self.weight = weight  # 损失的权重[15, 35, 1, 10]

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        # 强度损失
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_in_max)
        # 梯度损失
        ir_grad = self.sobelconv(self.bila(image_ir))  # 带双边滤波
        # ir_grad = self.sobelconv(image_ir)
        y_grad = self.sobelconv(image_y)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)

        loss_tra = F.l1_loss(generate_img, (image_ir+image_y)*0.5)
        # SSIM损失
        loss_ssim = self.L_SSIM(image_y, image_ir, generate_img)
        # 总损失
        loss_total = self.weight[0] * loss_in + self.weight[1] * loss_grad + \
                     self.weight[2] * loss_ssim + self.weight[3] * loss_tra
        return loss_total, loss_in, loss_grad, loss_ssim, loss_tra


if __name__ == '__main__':
    pass
