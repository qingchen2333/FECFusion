import datetime
import os
import random
import time
import pynvml
import numpy as np
import torch
import logging
from prettytable import PrettyTable
from FECFusion import FECFusion
from logger import setup_logger
from torch.autograd import Variable
from torch.utils.data import DataLoader
from loss import Fusionloss
from my_utils.dataset_util import Fusion_dataset
from my_utils.evaluation import eval_multi_method
from my_utils.my_args import get_train_args
from torch.utils.tensorboard import SummaryWriter
from my_utils.my_util import ColorSpaceTransform, RunningTime
from test import test


gpu_tamp = 78  # 控制GPU温度
weight = [10, 45, 0, 10]    # 强度最大，梯度最大，SSIM，内容
eval_flag = False    # 是否要指标评测
easy_flag = False


def init_seeds(args, seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def train():
    if eval_flag:
        table = PrettyTable(['Epoch', 'EN', 'MI', 'SF', 'AG', 'SD', 'CC', 'SCD',
                             'VIF', 'MSE', 'PSNR', 'Qabf', 'Nabf', 'SSIM', 'MS_SSIM'])
    start_time = time.time()
    # 一、初始化配置
    # 1.1 配置超参
    args = get_train_args()
    init_seeds(args)
    # 1.2 配置设备
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # print('Device is {}'.format(device))
    # 1.3 加载数据集
    train_dataset = Fusion_dataset('train', args.resize_flag, args.ir_path, args.vi_path)
    # print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=False
    )
    train_loader.n_iter = len(train_loader)
    # 1.4 加载网络模型
    ModelNet = FECFusion()
    if args.gpu >= 0:
        ModelNet.to(device)
    # 1.5 配置损失函数
    model_loss = Fusionloss(weight)
    # 1.6 配置优化器
    optimizer = torch.optim.Adam(ModelNet.parameters(), lr=args.learning_rate)
    # 1.7 添加tensorboard
    writer = SummaryWriter("./logs_train")
    # 1.8 配置logs消息日志
    log_path = './logs'
    logger = logging.getLogger()
    setup_logger(log_path)
    # 1.9 GPU温度监控
    pynvml.nvmlInit()  # 初始化
    # 获取GPU i的handle，后续通过handle来处理
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # 二、训练融合网络
    runtime = RunningTime()
    for epo in range(0, args.epoch):
        logger.info("--------------------------第{}轮训练--------------------------".format(epo + 1))

        # if epo < int(args.epoch // 2):  # epoch到总的一半的时候，学习率会随着epoch的增大而减小
        #     lr = args.learning_rate
        # else:
        #     lr = args.learning_rate * (args.epoch - epo) / (args.epoch - args.epoch // 2)
        if epo < (args.epoch // 2):  # epoch到总的一半的时候，学习率从0.001变成0.0001
            lr = args.learning_rate
        else:
            lr = 0.1*args.learning_rate
        # 修改学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 开始训练
        ModelNet.train()
        for it, (image_vis, image_ir, name, h, w) in enumerate(train_loader):
            image_vis = Variable(image_vis).to(device)
            image_vis_ycrcb = ColorSpaceTransform('RGB2YCrCb', image_vis)
            image_vis_y = image_vis_ycrcb[:, :1, :, :]
            image_ir = Variable(image_ir).to(device)
            optimizer.zero_grad()
            # forward
            logits = ModelNet(image_vis_y, image_ir)
            # loss
            loss_fusion, loss_in, loss_grad, loss_ssim, loss_tra = model_loss(image_vis_ycrcb, image_ir, logits)
            loss_fusion.backward()
            optimizer.step()
            # print loss
            eta, this_time, now_it = runtime.runtime(epo, it, train_loader.n_iter, args.epoch)
            if now_it % 100 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_fusion:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'loss_tra: {loss_tra:.4f}',
                        'lr: {lr:.4f}',
                        'time: {time:.2f}',
                        'eta: {eta}'
                    ]
                    ).format(
                        it=now_it,
                        max_it=train_loader.n_iter * args.epoch,
                        loss_fusion=loss_fusion.item(),
                        loss_in=loss_in.item(),
                        loss_grad=loss_grad.item(),
                        loss_ssim=loss_ssim.item(),
                        loss_tra=loss_tra.item(),
                        lr=lr,
                        time=this_time,
                        eta=eta
                    )
                logger.info(msg)
                writer.add_scalar("train_loss", loss_fusion.item(), now_it)

                # GPU温度检测
                gpuTemperature = pynvml.nvmlDeviceGetTemperature(handle, 0)  # 读取温度
                if gpuTemperature >= gpu_tamp:
                    print('GPU温度超过{}℃，开始降温！'.format(gpu_tamp))
                    time.sleep(5)  # 延时，让GPU温度没那么高

        # 三、保存模型权重
        if (epo + 1) >= 1:
            fusion_model_file = os.path.join(args.fusion_model_path, 'FusionModel{}.pth'.format(epo + 1))
            torch.save(ModelNet.state_dict(), fusion_model_file)
            logger.info("Fusion Model Save to: {}".format(fusion_model_file))
            if eval_flag:
                test((epo + 1), 'TNO')
                EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = \
                    eval_multi_method(easy_flag=easy_flag)
                val_list = [str((epo + 1)), round(EN, 4), round(MI, 4), round(SF, 4), round(AG, 4), round(SD, 4),
                               round(CC, 4), round(SCD, 4), round(VIF, 4), round(MSE, 4), round(PSNR, 4),
                               round(Qabf, 4), round(Nabf, 4), round(SSIM, 4), round(MS_SSIM, 4)]
                table.add_row(val_list)
                logger.info(val_list)
            elif (epo + 1) == args.epoch:
                test((epo + 1), 'TNO')

        if (epo + 1) >= args.epoch//2:
            test((epo + 1))

    end_time = time.time()
    all_time = int(end_time - start_time)
    eta = str(datetime.timedelta(seconds=all_time))
    logger.info('\n')
    logger.info('All train time is {}'.format(eta))
    if eval_flag:
        table.add_row(['SeAFusion', 7.1335, 2.833, 12.2525, 4.9803, 44.2436, 0.4819,    # 参考对比指标
                       1.7281, 0.7042, 0.059, 61.3917, 0.4879, 0.0807, 0.963, 0.9716])
        logger.info(table.get_string())     # 打印评价指标
    logger.info('weight = {}'.format(weight))


if __name__ == '__main__':
    train()
