import argparse


def get_test_args():
    parser = argparse.ArgumentParser(description='Test Image Fusion Model With Pytorch!')

    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_worker', '-W', type=int, default=0)  # 加载数据集的CPU线程数
    parser.add_argument('--fusion_model_path', '-S', type=str, default='./checkpoint')  # 模型权重路径
    parser.add_argument('--resize_flag', '-R', type=bool, default=False)
    parser.add_argument('--logs_path', type=str, default='./logs')  # 日志消息路径

    parser.add_argument('--ir_path1', type=str, default='./dataset/Test_ir/')  # 红外图像测试集路径
    parser.add_argument('--vi_path1', type=str, default='./dataset/Test_vi/')  # 可见光图像测试集路径
    parser.add_argument('--fused_path1', type=str, default='./results/FECFusion')  # 融化图像路径

    # parser.add_argument('--ir_path1', type=str, default=r'C:\Users\444\Desktop\seg/ir/')  # 红外图像测试集路径
    # parser.add_argument('--vi_path1', type=str, default=r'C:\Users\444\Desktop\seg/vi/')  # 可见光图像测试集路径
    # parser.add_argument('--fused_path1', type=str, default='./results/RepECFusion_seg')  # 融化图像路径

    parser.add_argument('--ir_path2', type=str, default=r'D:\czy\dataset\TNO/ir')  # 红外图像测试集路径
    parser.add_argument('--vi_path2', type=str, default=r'D:\czy\dataset\TNO/vi')  # 可见光图像测试集路径
    parser.add_argument('--fused_path2', type=str, default='./results/TNO')  # 融化图像路径

    test_args = parser.parse_args()
    return test_args


def get_train_args():
    parser = argparse.ArgumentParser(description='Train Image Fusion Model With Pytorch!')

    parser.add_argument('--batch_size', '-B', type=int, default=64)
    parser.add_argument('--epoch', '-E', type=int, default=4)
    parser.add_argument('--learning_rate', '-L', type=float, default=1e-3)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--resize_flag', '-R', type=bool, default=False)
    parser.add_argument('--num_worker', '-W', type=int, default=1)  # 加载数据集的CPU线程数
    parser.add_argument('--fusion_model_path', '-S', type=str, default='./checkpoint')
    parser.add_argument('--dataset_path', type=str, default='./dataset')  # 数据集路径
    parser.add_argument('--logs_path', type=str, default='./logs')  # 日志消息路径
    parser.add_argument('--ir_path', type=str, default='D:/czy/dataset/msrs_mini/Inf')  # 红外图像训练集路径
    parser.add_argument('--vi_path', type=str, default='D:/czy/dataset/msrs_mini/Vis')  # 可见光图像训练集路径

    train_args = parser.parse_args()
    return train_args


def get_ablation_args():
    parser = argparse.ArgumentParser(description='Test Image Fusion Model With Pytorch!')

    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_worker', '-W', type=int, default=0)  # 加载数据集的CPU线程数
    parser.add_argument('--fusion_model_path', '-S', type=str, default='./checkpoint')  # 模型权重路径
    parser.add_argument('--resize_flag', '-R', type=bool, default=False)
    parser.add_argument('--logs_path', type=str, default='./logs')  # 日志消息路径

    # parser.add_argument('--ir_path', type=str, default='./dataset/Test_ir/')    # 红外图像测试集路径
    # parser.add_argument('--vi_path', type=str, default='./dataset/Test_vi/')    # 可见光图像测试集路径
    # parser.add_argument('--fused_path', type=str, default='./results/REPFusion')  # 融化图像路径

    parser.add_argument('--ir_path1', type=str, default=r'D:\czy\Eval_dataset\TNO/ir')  # 红外图像测试集路径
    parser.add_argument('--vi_path1', type=str, default=r'D:\czy\Eval_dataset\TNO/vi')  # 可见光图像测试集路径
    parser.add_argument('--fused_path1', type=str, default='./results/FECFusion_TNO')  # 融化图像路径

    parser.add_argument('--ir_path2', type=str, default=r'D:\czy\Eval_dataset\RoadScene/ir')  # 红外图像测试集路径
    parser.add_argument('--vi_path2', type=str, default=r'D:\czy\Eval_dataset\RoadScene/vi')  # 可见光图像测试集路径
    parser.add_argument('--fused_path2', type=str, default='./results/FECFusion_RoadScene')  # 融化图像路径

    parser.add_argument('--ir_path3', type=str, default=r'D:\czy\Eval_dataset\MSRS/ir')  # 红外图像测试集路径
    parser.add_argument('--vi_path3', type=str, default=r'D:\czy\Eval_dataset\MSRS/vi')  # 可见光图像测试集路径
    parser.add_argument('--fused_path3', type=str, default='./results/FECFusion_MSRS')  # 融化图像路径

    parser.add_argument('--ir_path4', type=str, default=r'D:\czy\Eval_dataset\M3FD/ir')  # 红外图像测试集路径
    parser.add_argument('--vi_path4', type=str, default=r'D:\czy\Eval_dataset\M3FD/vi')  # 可见光图像测试集路径
    parser.add_argument('--fused_path4', type=str, default='./results/FECFusion_M3FD')  # 融化图像路径

    test_args = parser.parse_args()
    return test_args
