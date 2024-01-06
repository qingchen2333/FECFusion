import logging
import os

import numpy as np
import torch
from PIL import Image
import warnings
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from natsort import natsorted
from scipy.signal import convolve2d
import math
from scipy.fftpack import dctn
from scipy.ndimage import sobel, generic_gradient_magnitude
from torch import nn
from prettytable import PrettyTable
from tqdm import tqdm
from logger import setup_logger


warnings.filterwarnings("ignore")


def evaluation_one(ir_name, vi_name, f_name, easy_flag=False):
    f_img = Image.open(f_name).convert('L')
    ir_img = Image.open(ir_name).convert('L')
    vi_img = Image.open(vi_name).convert('L')
    f_img_int = np.array(f_img).astype(np.int32)

    f_img_double = np.array(f_img).astype(np.float32)
    ir_img_int = np.array(ir_img).astype(np.int32)
    ir_img_double = np.array(ir_img).astype(np.float32)

    vi_img_int = np.array(vi_img).astype(np.int32)
    vi_img_double = np.array(vi_img).astype(np.float32)

    EN = EN_function(f_img_int)
    MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)
    SF = SF_function(f_img_double)
    SD = SD_function(f_img_double)
    AG = AG_function(f_img_double)
    PSNR = PSNR_function(ir_img_double, vi_img_double, f_img_double)
    MSE = MSE_function(ir_img_double, vi_img_double, f_img_double)
    VIF = VIF_function(ir_img_double, vi_img_double, f_img_double)
    CC = CC_function(ir_img_double, vi_img_double, f_img_double)
    SCD = SCD_function(ir_img_double, vi_img_double, f_img_double)
    Qabf = Qabf_function(ir_img_double, vi_img_double, f_img_double)
    if easy_flag:
        Nabf, SSIM, MS_SSIM = 0.0, 0.0, 0.0
    else:
        Nabf = Nabf_function(ir_img_double, vi_img_double, f_img_double)
        SSIM = SSIM_function(ir_img_double, vi_img_double, f_img_double)
        MS_SSIM = MS_SSIM_function(ir_img_double, vi_img_double, f_img_double)

    return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM


def EN_function(image_array):
    # 计算图像的直方图
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 255))
    # 将直方图归一化
    histogram = histogram / float(np.sum(histogram))
    # 计算熵
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy


def SF_function(image):
    image_array = np.array(image)
    RF = np.diff(image_array, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(image_array, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF


def SD_function(image_array):
    m, n = image_array.shape
    u = np.mean(image_array)
    SD = np.sqrt(np.sum(np.sum((image_array - u) ** 2)) / (m * n))
    return SD


def PSNR_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * np.log10(255 / np.sqrt(MSE))
    return PSNR


def MSE_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE


def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian((N, N), N / 5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist * dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref * dist, win, mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    vifp = num / den
    return vifp


def VIF_function(A, B, F):
    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF


def CC_function(A, B, F):
    rAF = np.sum((A - np.mean(A)) * (F - np.mean(F))) / np.sqrt(
        np.sum((A - np.mean(A)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    rBF = np.sum((B - np.mean(B)) * (F - np.mean(F))) / np.sqrt(
        np.sum((B - np.mean(B)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    CC = np.mean([rAF, rBF])
    return CC


def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r


def SCD_function(A, B, F):
    r = corr2(F - B, A) + corr2(F - A, B)
    return r


def Qabf_function(A, B, F):
    return get_Qabf(A, B, F)


def Nabf_function(A, B, F):
    return Nabf_function(A, B, F)


def Hab(im1, im2, gray_level):
    hang, lie = im1.shape
    count = hang * lie
    N = gray_level
    h = np.zeros((N, N))
    for i in range(hang):
        for j in range(lie):
            h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
    h = h / np.sum(h)
    im1_marg = np.sum(h, axis=0)
    im2_marg = np.sum(h, axis=1)
    H_x = 0
    H_y = 0
    for i in range(N):
        if (im1_marg[i] != 0):
            H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])
    for i in range(N):
        if (im2_marg[i] != 0):
            H_x = H_x + im2_marg[i] * math.log2(im2_marg[i])
    H_xy = 0
    for i in range(N):
        for j in range(N):
            if (h[i, j] != 0):
                H_xy = H_xy + h[i, j] * math.log2(h[i, j])
    MI = H_xy - H_x - H_y
    return MI


def MI_function(A, B, F, gray_level=256):
    MIA = Hab(A, F, gray_level)
    MIB = Hab(B, F, gray_level)
    MI_results = MIA + MIB
    return MI_results


def AG_function(image):
    width = image.shape[1]
    width = width - 1
    height = image.shape[0]
    height = height - 1
    tmp = 0.0
    [grady, gradx] = np.gradient(image)
    s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
    AG = np.sum(np.sum(s)) / (width * height)
    return AG


def SSIM_function(A, B, F):
    ssim_A = ssim(A, F)
    ssim_B = ssim(B, F)
    SSIM = 1 * ssim_A + 1 * ssim_B
    return SSIM.item()


def MS_SSIM_function(A, B, F):
    ssim_A = ms_ssim(A, F)
    ssim_B = ms_ssim(B, F)
    MS_SSIM = 1 * ssim_A + 1 * ssim_B
    return MS_SSIM.item()


def Nabf_function(A, B, F):
    Nabf = get_Nabf(A, B, F)
    return Nabf


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            perms = list(range(win.ndim))
            perms[2 + i] = perms[-1]
            perms[-1] = 2 + i
            out = conv(out, weight=win.permute(perms), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.type_as(X)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(X,
         Y,
         data_range=255,
         size_average=True,
         win_size=11,
         win_sigma=1.5,
         win=None,
         K=(0.01, 0.03),
         nonnegative_ssim=False):
    # 输出的是灰度图像，其shape是[H, W]
    # 需要扩展为 [B, C, H, W]
    X = TF.to_tensor(X).unsqueeze(0).unsqueeze(0) * 255.0
    Y = TF.to_tensor(Y).unsqueeze(0).unsqueeze(0) * 255.0
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = torch.squeeze(X, dim=d)
        Y = torch.squeeze(Y, dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, _ = _ssim(X, Y, data_range=data_range, win=win, K=K)
    if nonnegative_ssim:
        ssim_per_channel = F.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(dim=1)


def ms_ssim(
        X,
        Y,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        win=None,
        weights=None,
        K=(0.01, 0.03)
):
    # 输出的是灰度图像，其shape是[H, W]
    # 需要扩展为 [B, C, H, W]
    X = TF.to_tensor(X).unsqueeze(0).unsqueeze(0) * 255.0
    Y = TF.to_tensor(Y).unsqueeze(0).unsqueeze(0) * 255.0
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
            2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.tensor(weights, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, K=K)

        if i < levels - 1:
            mcs.append(F.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = F.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.reshape((-1, 1, 1)), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(dim=1)


class SSIM(nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
    ):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).tile([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        ).item()


class MS_SSIM(nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            weights=None,
            K=(0.01, 0.03),
    ):
        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).tile([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        ).item()


def sobel_fn(x):
    # Sobel operators
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)
    p, q = x_ext.shape
    gv = np.zeros((p - 2, q - 2))
    gh = np.zeros((p - 2, q - 2))
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')
    # for ii in range(1, p - 1):
    #     for jj in range(1, q - 1):
    #         gv[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * vtemp)
    #         gh[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * htemp)

    return gv, gh


def per_extn_im_fn(x, wsize):
    """
    Periodic extension of the given image in 4 directions.

    xout_ext = per_extn_im_fn(x, wsize)

    Periodic extension by (wsize-1)/2 on all 4 sides.
    wsize should be odd.

    Example:
        Y = per_extn_im_fn(X, 5);    % Periodically extends 2 rows and 2 columns in all sides.
    """

    hwsize = (wsize - 1) // 2  # Half window size excluding centre pixel.

    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # Row-wise periodic extension.
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    # Column-wise periodic extension.
    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext


def get_Nabf(I1, I2, f):
    # Parameters for Petrovic Metrics Computation.
    Td = 2
    wt_min = 0.001
    P = 1
    Lg = 1.5
    Nrg = 0.9999
    kg = 19
    sigmag = 0.5
    Nra = 0.9995
    ka = 22
    sigmaa = 0.5

    xrcw = f.astype(np.float64)
    x1 = I1.astype(np.float64)
    x2 = I2.astype(np.float64)

    # Edge Strength & Orientation.
    gvA, ghA = sobel_fn(x1)
    gA = np.sqrt(ghA ** 2 + gvA ** 2)

    gvB, ghB = sobel_fn(x2)
    gB = np.sqrt(ghB ** 2 + gvB ** 2)

    gvF, ghF = sobel_fn(xrcw)
    gF = np.sqrt(ghF ** 2 + gvF ** 2)

    # Relative Edge Strength & Orientation.
    gAF = np.zeros(gA.shape)
    gBF = np.zeros(gB.shape)
    aA = np.zeros(ghA.shape)
    aB = np.zeros(ghB.shape)
    aF = np.zeros(ghF.shape)
    p, q = xrcw.shape
    maskAF1 = (gA == 0) | (gF == 0)
    maskAF2 = (gA > gF)
    gAF[~maskAF1] = np.where(maskAF2, gF / gA, gA / gF)[~maskAF1]
    maskBF1 = (gB == 0) | (gF == 0)
    maskBF2 = (gB > gF)
    gBF[~maskBF1] = np.where(maskBF2, gF / gB, gB / gF)[~maskBF1]
    aA = np.where((gvA == 0) & (ghA == 0), 0, np.arctan(gvA / ghA))
    aB = np.where((gvB == 0) & (ghB == 0), 0, np.arctan(gvB / ghB))
    aF = np.where((gvF == 0) & (ghF == 0), 0, np.arctan(gvF / ghF))

    aAF = np.abs(np.abs(aA - aF) - np.pi / 2) * 2 / np.pi
    aBF = np.abs(np.abs(aB - aF) - np.pi / 2) * 2 / np.pi

    QgAF = Nrg / (1 + np.exp(-kg * (gAF - sigmag)))
    QaAF = Nra / (1 + np.exp(-ka * (aAF - sigmaa)))
    QAF = np.sqrt(QgAF * QaAF)
    QgBF = Nrg / (1 + np.exp(-kg * (gBF - sigmag)))
    QaBF = Nra / (1 + np.exp(-ka * (aBF - sigmaa)))
    QBF = np.sqrt(QgBF * QaBF)

    wtA = wt_min * np.ones((p, q))
    wtB = wt_min * np.ones((p, q))
    cA = np.ones((p, q))
    cB = np.ones((p, q))
    wtA = np.where(gA >= Td, cA * gA ** Lg, 0)
    wtB = np.where(gB >= Td, cB * gB ** Lg, 0)

    wt_sum = np.sum(wtA + wtB)
    QAF_wtsum = np.sum(QAF * wtA) / wt_sum  # Information Contributions of A.
    QBF_wtsum = np.sum(QBF * wtB) / wt_sum  # Information Contributions of B.
    QABF = QAF_wtsum + QBF_wtsum  # QABF=sum(sum(QAF.*wtA+QBF.*wtB))/wt_sum -> Total Fusion Performance.

    Qdelta = np.abs(QAF - QBF)
    QCinfo = (QAF + QBF - Qdelta) / 2
    QdeltaAF = QAF - QCinfo
    QdeltaBF = QBF - QCinfo
    QdeltaAF_wtsum = np.sum(QdeltaAF * wtA) / wt_sum
    QdeltaBF_wtsum = np.sum(QdeltaBF * wtB) / wt_sum
    QdeltaABF = QdeltaAF_wtsum + QdeltaBF_wtsum  # Total Fusion Gain.
    QCinfo_wtsum = np.sum(QCinfo * (wtA + wtB)) / wt_sum
    QABF11 = QdeltaABF + QCinfo_wtsum  # Total Fusion Performance.

    rr = np.zeros((p, q))
    rr = np.where(gF <= np.minimum(gA, gB), 1, 0)

    LABF = np.sum(rr * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

    na1 = np.where((gF > gA) & (gF > gB), 2 - QAF - QBF, 0)
    NABF1 = np.sum(na1 * (wtA + wtB)) / wt_sum

    # Fusion Artifacts (NABF) changed by B. K. Shreyamsha Kumar.

    na = np.where((gF > gA) & (gF > gB), 1, 0)
    NABF = np.sum(na * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum
    return NABF


def sobel_fn(x):
    # Sobel operators
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)
    p, q = x_ext.shape
    gv = np.zeros((p - 2, q - 2))
    gh = np.zeros((p - 2, q - 2))
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')
    # for ii in range(1, p - 1):
    #     for jj in range(1, q - 1):
    #         gv[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * vtemp)
    #         gh[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * htemp)

    return gv, gh


def per_extn_im_fn(x, wsize):
    """
    Periodic extension of the given image in 4 directions.

    xout_ext = per_extn_im_fn(x, wsize)

    Periodic extension by (wsize-1)/2 on all 4 sides.
    wsize should be odd.

    Example:
        Y = per_extn_im_fn(X, 5);    % Periodically extends 2 rows and 2 columns in all sides.
    """

    hwsize = (wsize - 1) // 2  # Half window size excluding centre pixel.

    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # Row-wise periodic extension.
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    # Column-wise periodic extension.
    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext

def get_Qabf(pA, pB, pF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5;
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    # if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
    # 如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

    strA = pA
    strB = pB
    strF = pF

    # 数组旋转180度
    def flip180(arr):
        return np.flip(arr)

    # 相当于matlab的Conv2
    def convolution(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        img_new = convolve2d(data, k, mode='valid')
        return img_new

    def getArray(img):
        SAx = convolution(h3, img)
        SAy = convolution(h1, img)
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        n, m = img.shape
        aA = np.zeros((n, m))
        zero_mask = SAx == 0
        aA[~zero_mask] = np.arctan(SAy[~zero_mask] / SAx[~zero_mask])
        aA[zero_mask] = np.pi / 2
        # for i in range(n):
        #     for j in range(m):
        #         if (SAx[i, j] == 0):
        #             aA[i, j] = math.pi / 2
        #         else:
        #             aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
        return gA, aA

    # 对strB和strF进行相同的操作
    gA, aA = getArray(strA)
    gB, aB = getArray(strB)
    gF, aF = getArray(strF)

    # the relative strength and orientation value of GAF,GBF and AAF,ABF;
    def getQabf(aA, gA, aF, gF):
        mask = (gA > gF)
        GAF = np.where(mask, gF / gA, np.where(gA == gF, gF, gA / gF))

        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)

        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))

        QAF = QgAF * QaAF
        return QAF

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    output = nume / deno
    return output


def analysis_fmi(ima, imb, imf, feature, w):
    ima = np.double(ima)
    imb = np.double(imb)
    imf = np.double(imf)

    # Feature Extraction
    if feature == 'none':  # Raw pixels (no feature extraction)
        aFeature = ima
        bFeature = imb
        fFeature = imf
    elif feature == 'gradient':  # Gradient
        aFeature = generic_gradient_magnitude(ima, sobel)
        bFeature = generic_gradient_magnitude(imb, sobel)
        fFeature = generic_gradient_magnitude(imf, sobel)
    elif feature == 'edge':  # Edge
        aFeature = np.double(sobel(ima) > w)
        bFeature = np.double(sobel(imb) > w)
        fFeature = np.double(sobel(imf) > w)
    elif feature == 'dct':  # DCT
        aFeature = dctn(ima, type=2, norm='ortho')
        bFeature = dctn(imb, type=2, norm='ortho')
        fFeature = dctn(imf, type=2, norm='ortho')
    elif feature == 'wavelet':  # Discrete Meyer wavelet
        raise NotImplementedError('Wavelet feature extraction not yet implemented in Python!')
    else:
        raise ValueError(
            "Please specify a feature extraction method among 'gradient', 'edge', 'dct', 'wavelet', or 'none' (raw "
            "pixels)!")

    m, n = aFeature.shape
    w = w // 2
    fmi_map = np.ones((m - 2 * w, n - 2 * w))


def eval_multi_method(eval_flag=None, dataroot=None, results_root=None, dataset=None, easy_flag=False):
    if dataroot is None:
        dataroot = r'D:/czy/dataset'
    if results_root is None:
        results_root = r'D:/czy/FEFusion_czy/results/'
    if dataset is None:
        dataset = 'TNO'
    ir_dir = os.path.join(dataroot, dataset, 'ir')
    vi_dir = os.path.join(dataroot, dataset, 'vi')
    f_dir = os.path.join(results_root, dataset)
    filelist = natsorted(os.listdir(ir_dir))
    if eval_flag is None:   # 如果是正式评估，需要 eval_flag=True
        file_list = []
        for i in range(20):     # 42张图太久了，只要前20张
            file_list.append(filelist[i])
    else:
        file_list = filelist

    EN_list = []
    MI_list = []
    SF_list = []
    AG_list = []
    SD_list = []
    CC_list = []
    SCD_list = []
    VIF_list = []
    MSE_list = []
    PSNR_list = []
    Qabf_list = []
    Nabf_list = []
    SSIM_list = []
    MS_SSIM_list = []

    sub_f_dir = os.path.join(f_dir)
    eval_bar = tqdm(file_list)
    for _, item in enumerate(eval_bar):
        ir_name = os.path.join(ir_dir, item)
        vi_name = os.path.join(vi_dir, item)
        f_name = os.path.join(sub_f_dir, item)
        # print(ir_name, vi_name, f_name)
        EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir_name, vi_name,
                                                                                                f_name, easy_flag)
        EN_list.append(EN)
        MI_list.append(MI)
        SF_list.append(SF)
        AG_list.append(AG)
        SD_list.append(SD)
        CC_list.append(CC)
        SCD_list.append(SCD)
        VIF_list.append(VIF)
        MSE_list.append(MSE)
        PSNR_list.append(PSNR)
        Qabf_list.append(Qabf)
        Nabf_list.append(Nabf)
        SSIM_list.append(SSIM)
        MS_SSIM_list.append(MS_SSIM)

        eval_bar.set_description("Eval | {}".format(item))

    # 均值
    EN_mean = np.mean(EN_list)
    MI_mean = np.mean(MI_list)
    SF_mean = np.mean(SF_list)
    AG_mean = np.mean(AG_list)
    SD_mean = np.mean(SD_list)
    CC_mean = np.mean(CC_list)
    SCD_mean = np.mean(SCD_list)
    VIF_mean = np.mean(VIF_list)
    MSE_mean = np.mean(MSE_list)
    PSNR_mean = np.mean(PSNR_list)
    Qabf_mean = np.mean(Qabf_list)
    Nabf_mean = np.mean(Nabf_list)
    SSIM_mean = np.mean(SSIM_list)
    MS_SSIM_mean = np.mean(MS_SSIM_list)
    return EN_mean, MI_mean, SF_mean, AG_mean, SD_mean, CC_mean, SCD_mean, VIF_mean, \
           MSE_mean, PSNR_mean, Qabf_mean, Nabf_mean, SSIM_mean, MS_SSIM_mean


if __name__ == '__main__':
    algorithms = ['DenseFuse', 'RFN-Nest', 'FusionGAN', 'IFCNN', 'PMGI', 'SDNet',
                  'U2Fusion', 'FLFuse', 'SeAFusion', 'PIAFusion', 'FEFusion']
    datasets = ['TNO', 'RoadScene', 'MSRS', 'M3FD']         # 'TNO', 'RoadScene', 'MSRS', 'M3FD'
    dataroot = r'D:\czy\Eval_dataset'    # 数据集路径
    results_root = r'D:\czy\All_Results'    # 算法结果路径

    log_path = '../logs'
    logger = logging.getLogger()
    setup_logger(log_path)

    for i in range(len(datasets)):
        dataset = datasets[i]
        logger.info('Dataset: {}'.format(dataset))
        table = PrettyTable(['Algorithm', 'EN', 'MI', 'SF', 'AG', 'SD', 'CC', 'SCD',
                             'VIF', 'MSE', 'PSNR', 'Qabf', 'Nabf', 'SSIM', 'MS_SSIM'])
        for j in range(len(algorithms)):
            algorithm = algorithms[j]
            logger.info('Algorithm: {}'.format(algorithm))
            fused_path = os.path.join(results_root, algorithm)
            EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = \
                eval_multi_method(True, dataroot, fused_path, dataset)

            table.add_row([str(algorithm), round(EN, 4), round(MI, 4), round(SF, 4), round(AG, 4), round(SD, 4),
                           round(CC, 4), round(SCD, 4), round(VIF, 4), round(MSE, 4), round(PSNR, 4),
                           round(Qabf, 4), round(Nabf, 4), round(SSIM, 4), round(MS_SSIM, 4)])
        logger.info(table.get_string())
