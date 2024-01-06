# coding:utf-8
import torch
import torch.nn as nn
from torchsummary import summary
from ecb import ECB


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        x = self.sigmoid(max_out + avg_out) * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1))) * x
        return x


# 完整主模型
class FECFusion(nn.Module):
    def __init__(self, deploy=False):
        super(FECFusion, self).__init__()
        ch = [1, 16, 32, 64, 128]
        self.act_type = 'lrelu'
        self.conv0_vi = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)
        self.conv0_ir = nn.Conv2d(ch[0], ch[1], kernel_size=1, padding=0)
        self.conv1_vi = ECB(ch[1], ch[2], 2, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv1_ir = ECB(ch[1], ch[2], 2, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2_vi = ECB(ch[2], ch[3], 2, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2_ir = ECB(ch[2], ch[3], 2, act_type=self.act_type, with_idt=False, deploy=deploy)

        self.CBAM1 = CBAMLayer(ch[3])
        self.CBAM2 = CBAMLayer(ch[3])

        self.conv1 = ECB(ch[4], ch[3], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv2 = ECB(ch[3], ch[2], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv3 = ECB(ch[2], ch[1], 0.5, act_type=self.act_type, with_idt=False, deploy=deploy)
        self.conv4 = nn.Conv2d(ch[1], ch[0], kernel_size=1, padding=0)

        self.act = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, image_vi, image_ir):
        # encoder
        x_vi = self.act(self.conv0_vi(image_vi))
        x_ir = self.act(self.conv0_ir(image_ir))
        x_vi = self.conv1_vi(x_vi)
        x_ir = self.conv1_ir(x_ir)
        x_vi = self.conv2_vi(x_vi)
        x_ir = self.conv2_ir(x_ir)

        # fusion
        x = torch.cat([(x_vi * x_ir), (self.CBAM1(x_vi) + self.CBAM2(x_ir))], dim=1)

        # decoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.tanh(self.conv4(x))
        return x / 2 + 0.5


def model_deploy(model):
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    return model


def unit_test():
    import time
    n = 100  # 循环次数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 1, 480, 640).cuda()
    model = FECFusion().to(device)
    model.eval()
    for i in range(10):
        train_y = model(x, x)    # 第一次运行的不是准确的时间，可能还要加载模型之类的操作

    start_time = time.time()
    for i in range(n):
        train_y = model(x, x)
    train_y_time = time.time() - start_time

    print(summary(model, [(1, 480, 640), (1, 480, 640)]))
    model = model_deploy(model)
    print(summary(model, [(1, 480, 640), (1, 480, 640)]))

    for i in range(10):
        train_y = model(x, x)    # 第一次运行的不是准确的时间，可能还要加载模型之类的操作

    start_time = time.time()
    for i in range(n):
        deploy_y = model(x, x)
    deploy_y_time = time.time() - start_time

    print('train__y time is {:.4f}s/it'.format(train_y_time/n))
    print('deploy_y time is {:.4f}s/it'.format(deploy_y_time/n))
    print('The different is', (train_y - deploy_y).sum())


if __name__ == '__main__':
    unit_test()
