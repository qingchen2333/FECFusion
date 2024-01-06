# coding:utf-8
import os
import time

import torch
from natsort import natsorted
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import glob
import os


def prepare_data_path(dataset_path):
    filenames = natsorted(os.listdir(dataset_path))
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data = natsorted(data)
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, resize_flag, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'test'], 'split must be "train"|"test"'
        # 为了方便有些模型是需要标签或者别的不同训练集，因此分开写，有用到专用的标签或别的时候就专门去写路径
        self.h = 512
        self.w = 512
        if split == 'train':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.resize_flag = resize_flag
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'test':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.resize_flag = resize_flag
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            if self.resize_flag:
                image_vis = np.array(Image.open(vis_path).resize((320, 240), Image.ANTIALIAS))
                image_inf = cv2.imread(ir_path, 0)
                self.h, self.w = image_inf.shape
                image_inf = cv2.resize(image_inf, (320, 240), interpolation=cv2.INTER_CUBIC)
            else:
                image_vis = np.array(Image.open(vis_path))
                image_inf = cv2.imread(ir_path, 0)
                self.h, self.w = image_inf.shape

            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
                self.h,
                self.w
            )
        elif self.split == 'test':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            if self.resize_flag:
                image_vis = np.array(Image.open(vis_path).resize((512, 512), Image.ANTIALIAS))
                image_inf = cv2.imread(ir_path, 0)
                self.h, self.w = image_inf.shape
                image_inf = cv2.resize(image_inf, (512, 512), interpolation=cv2.INTER_CUBIC)
            else:
                image_vis = np.array(Image.open(vis_path))
                image_inf = cv2.imread(ir_path, 0)
                self.h, self.w = image_inf.shape

            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
                self.h,
                self.w
            )

    def __len__(self):
        return self.length


# 测试加入mask
# class Fusion_dataset(Dataset):
#     def __init__(self, split, resize_flag, ir_path=None, vi_path=None, mask_path=None):
#         super(Fusion_dataset, self).__init__()
#         assert split in ['train', 'test'], 'split must be "train"|"test"'
#         # 为了方便有些模型是需要标签或者别的不同训练集，因此分开写，有用到专用的标签或别的时候就专门去写路径
#         if split == 'train':
#             data_dir_vis = vi_path
#             data_dir_ir = ir_path
#             data_dir_mask = mask_path
#             self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
#             self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
#             self.filepath_mask, self.filenames_mask = prepare_data_path(data_dir_mask)
#             self.split = split
#             self.resize_flag = resize_flag
#             self.length = min(len(self.filenames_vis), len(self.filenames_ir))
#
#         elif split == 'test':
#             data_dir_vis = vi_path
#             data_dir_ir = ir_path
#             data_dir_mask = mask_path
#             self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
#             self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
#             self.filepath_mask, self.filenames_mask = prepare_data_path(data_dir_mask)
#             self.split = split
#             self.resize_flag = resize_flag
#             self.length = min(len(self.filenames_vis), len(self.filenames_ir))
#
#     def __getitem__(self, index):
#         if self.split == 'train':
#             vis_path = self.filepath_vis[index]
#             ir_path = self.filepath_ir[index]
#             mask_path = self.filepath_mask[index]
#             if self.resize_flag:
#                 # image_vis = np.array(Image.open(vis_path).resize((64, 48), Image.ANTIALIAS))
#                 image_vis = cv2.imread(vis_path, 0)
#                 image_vis = cv2.resize(image_vis, (360, 270), interpolation=cv2.INTER_CUBIC)
#                 image_inf = cv2.imread(ir_path, 0)
#                 image_inf = cv2.resize(image_inf, (360, 270), interpolation=cv2.INTER_CUBIC)
#                 image_mask = cv2.imread(mask_path, 0)
#                 image_mask = cv2.resize(image_mask, (360, 270), interpolation=cv2.INTER_CUBIC)
#             else:
#                 image_vis = cv2.imread(vis_path, 0)  # np.array(Image.open(vis_path))
#                 image_inf = cv2.imread(ir_path, 0)
#                 image_mask = cv2.imread(mask_path, 0)
#
#             # image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0)
#             image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
#             image_vis = np.expand_dims(image_vis, axis=0)
#             image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
#             image_ir = np.expand_dims(image_ir, axis=0)
#             image_mask = np.asarray(Image.fromarray(image_mask), dtype=np.float32) / 255.0
#             image_mask = np.expand_dims(image_mask, axis=0)
#             name = self.filenames_vis[index]
#             return (
#                 torch.tensor(image_vis),
#                 torch.tensor(image_ir),
#                 torch.tensor(image_mask),
#                 name
#             )
#         elif self.split == 'test':
#             vis_path = self.filepath_vis[index]
#             ir_path = self.filepath_ir[index]
#             mask_path = self.filepath_mask[index]
#             # image_vis = np.array(Image.open(vis_path))
#             image_vis = cv2.imread(vis_path, 0)
#             image_inf = cv2.imread(ir_path, 0)
#             image_mask = cv2.imread(mask_path, 0)
#             # image_vis = (
#             #     np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0
#             # )
#             image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
#             image_vis = np.expand_dims(image_vis, axis=0)
#             image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
#             image_ir = np.expand_dims(image_ir, axis=0)
#             image_mask = np.asarray(Image.fromarray(image_mask), dtype=np.float32) / 255.0
#             image_mask = np.expand_dims(image_mask, axis=0)
#             name = self.filenames_vis[index]
#             return (
#                 torch.tensor(image_vis),
#                 torch.tensor(image_ir),
#                 torch.tensor(image_mask),
#                 name
#             )
#
#     def __len__(self):
#         return self.length


# if __name__ == '__main__':
# data_dir = '/data1/yjt/MFFusion/dataset/'
# train_dataset = MF_dataset(data_dir, 'train', have_label=True)
# print("the training dataset is length:{}".format(train_dataset.length))
# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=2,
#     shuffle=True,
#     num_workers=2,
#     pin_memory=True,
#     drop_last=True,
# )
# train_loader.n_iter = len(train_loader)
# for it, (image_vis, image_ir, label) in enumerate(train_loader):
#     if it == 5:
#         image_vis.numpy()
#         print(image_vis.shape)
#         image_ir.numpy()
#         print(image_ir.shape)
#         break
