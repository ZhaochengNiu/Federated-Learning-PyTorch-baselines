#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) 2022  Gabriele Cazzato

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

import torch
# 导入PyTorch库，这是一个广泛使用的开源机器学习库，特别适合处理基于GPU的操作。
from torch import nn
# 从PyTorch库中导入nn模块，它包含构建神经网络所需的类和函数。
import torchvision.models as tvmodels
# 导入torchvision.models模块作为tvmodels，这个模块提供了多种预训练的模型架构。
from torchvision.transforms import Resize
# 从torchvision.transforms模块中导入Resize类，用于对图像进行尺寸调整的变换。
from ghostnet import ghostnet as load_ghostnet
# 导入ghostnet模块，并将其别名设置为load_ghostnet。
# 这可能是一个自定义模块，用于加载GhostNet模型，这是一种轻量级的深度学习架构。
#from tinynet import tinynet as load_tinynet
from models_utils import *
# 从models_utils模块导入所有内容，这可能是一个包含自定义模型工具和实用函数的模块。


# From "Communication-Efficient Learning of Deep Networks from Decentralized Data"
class mlp_mnist(nn.Module):
    # 定义了一个名为 mlp_mnist 的类，它继承自 PyTorch 的 nn.Module 类，这是所有神经网络模块的基类。
    def __init__(self, num_classes, num_channels, model_args):
        # 构造函数接收三个参数：
        # num_classes：输出类别的数量，通常用于多类分类任务。
        # num_channels：输入图像的通道数，对于 MNIST 数据集，通常是 1（灰度图像）。
        # model_args：模型参数的字典，可能包含其他用于初始化模型的参数。
        super(mlp_mnist, self).__init__()
        # 调用父类 nn.Module 的构造函数，这是 Python 中类的初始化的惯用方法。
        self.resize = Resize((28, 28))
        # 初始化一个 Resize 变换，将输入图像调整为 28x28 像素。这通常用于确保输入图像的尺寸一致。
        # 初始化一个 nn.Sequential 容器，它将按照顺序包含多个神经网络层。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 添加一个 Flatten 层，将多维输入张量展平为一维。
            nn.Linear(num_channels*28*28, 200),
            # 添加一个全连接层，将展平后的输入特征从 num_channels*28*28 维映射到 200 维。
            nn.ReLU(),
            # 添加一个 ReLU 激活函数层，为网络引入非线性。
            nn.Linear(200, 200),
            # 再次添加一个全连接层，将 200 维特征映射回 200 维，这通常用于增加网络的非线性能力。
            nn.ReLU(),
            # 再次添加一个 ReLU 激活函数层。
            nn.Linear(200, num_classes),
            # 最后添加一个全连接层，将 200 维特征映射到 num_classes 维，对应于输出类别的数量。
        )
    def forward(self, x):
        # 定义 forward 方法，它是每个 nn.Module 子类必须实现的方法，用于指定如何计算前向传播。
        x = self.resize(x)
        # 将输入 x 通过 Resize 变换调整尺寸。
        x = self.classifier(x)
        # 将调整尺寸后的输入 x 通过 classifier 网络进行前向传播。
        return x


# From "Communication-Efficient Learning of Deep Networks from Decentralized Data"
class cnn_mnist(nn.Module):
    # 定义了一个名为 cnn_mnist 的类，用于构建针对 MNIST 数据集的卷积神经网络。
    def __init__(self, num_classes, num_channels, model_args):
        # 构造函数接收三个参数：
        # num_classes：输出类别的数量，MNIST 数据集有 10 个类别（0 到 9 的数字）。
        # num_channels：输入图像的通道数，对于 MNIST，通常是 1（灰度图像）。
        # model_args：模型参数的字典，可能包含其他用于初始化模型的参数。
        super(cnn_mnist, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        self.resize = Resize((28, 28))
        # 初始化一个 Resize 变换，确保输入图像的尺寸统一为 28x28 像素。
        self.feature_extractor = nn.Sequential(
            # 初始化一个 nn.Sequential 容器，用于特征提取，包含多个按顺序执行的层。
            nn.Conv2d(num_channels, 32, kernel_size=5, stride=1, padding=1),
            # 第一个卷积层，从 num_channels 个输入通道映射到 32 个输出通道，使用 5x5 的卷积核，步长为 1，填充为 1。
            nn.ReLU(),
            # ReLU 激活函数层。
            nn.MaxPool2d(2, stride=2, padding=1),
            # 最大池化层，使用 2x2 的池化窗口，步长为 2，填充为 1。
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            # 第二个卷积层，从 32 个输入通道映射到 64 个输出通道，使用 5x5 的卷积核，步长为 1，填充为 1。
            nn.ReLU(),
            # 第二个 ReLU 激活函数层。
            nn.MaxPool2d(2, stride=2, padding=1),
            # 第二个最大池化层，使用 2x2 的池化窗口，步长为 2，填充为 1。
        )

        self.classifier = nn.Sequential(
            # 初始化一个 nn.Sequential 容器，用于分类器，包含全连接层。
            nn.Flatten(),
            # Flatten 层，将多维输入张量展平为一维。
            nn.Linear(64*7*7, 512),
            # 第一个全连接层，将展平后的输入特征从 6477 维映射到 512 维。这里的 7*7 来自两次池化后的特征图尺寸。
            nn.ReLU(),
            # 第二个 ReLU 激活函数层。
            nn.Linear(512, num_classes),
            # 第二个全连接层，将 512 维特征映射到 num_classes 维，对应于输出类别的数量
        )

    def forward(self, x):
        # 定义 forward 方法，用于指定模型的前向传播过程。
        x = self.resize(x)
        # 将输入 x 通过 Resize 变换调整尺寸。
        x = self.feature_extractor(x)
        # 将调整尺寸后的输入 x 通过特征提取器进行卷积和池化操作。
        x = self.classifier(x)
        # 将特征提取器的输出通过分类器进行全连接层操作。
        return x
        # 返回模型的输出，即分类的 logits 或概率。


# From "Communication-Efficient Learning of Deep Networks from Decentralized Data"
# (ported from 2016 TensorFlow CIFAR-10 tutorial)
class cnn_cifar10(nn.Module):
    # 定义了一个名为 cnn_cifar10 的类，用于构建针对 CIFAR-10 数据集的卷积神经网络。
    def __init__(self, num_classes, num_channels, model_args):
        # 构造函数接收三个参数：
        # num_classes：输出类别的数量，CIFAR-10 数据集有 10 个类别。
        # num_channels：输入图像的通道数，CIFAR-10 图像是 RGB 彩色图像，因此通常是 3。
        # model_args：模型参数的字典，可能包含其他用于初始化模型的参数。
        super(cnn_cifar10, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        self.resize = Resize((24, 24))
        # 初始化一个 Resize 变换，将输入图像调整为 24x24 像素。CIFAR-10 图像原始尺寸为 32x32 像素。
        self.feature_extractor = nn.Sequential(
            # 初始化一个 nn.Sequential 容器，用于特征提取，包含多个按顺序执行的层。
            nn.Conv2d(num_channels, 64, kernel_size=5, stride=1, padding='same'),
            # 第一个卷积层，从 num_channels 个输入通道映射到 64 个输出通道，
            # 使用 5x5 的卷积核，步长为 1，填充方式为 'same'，即输出尺寸与输入尺寸相同。
            nn.ReLU(),
            # ReLU 激活函数层。
            nn.ZeroPad2d((0, 1, 0, 1)), # Equivalent of TensorFlow padding 'SAME' for MaxPool2d
            # 零填充层，用于在卷积后保持特征图尺寸，以匹配 TensorFlow 中 'SAME' 填充的效果。
            nn.MaxPool2d(3, stride=2, padding=0),
            # 最大池化层，使用 3x3 的池化窗口，步长为 2，填充为 0。
            nn.LocalResponseNorm(4, alpha=0.001/9),
            # 局部响应归一化层，用于归一化相邻的特征。
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
            # 第二个卷积层，将 64 个输入通道映射到 64 个输出通道，使用 5x5 的卷积核，步长为 1，填充方式为 'same'。
            nn.ReLU(),
            # 第二个 ReLU 激活函数层。
            nn.LocalResponseNorm(4, alpha=0.001/9),
            # 第二个局部响应归一化层。
            nn.ZeroPad2d((0, 1, 0, 1)), # Equivalent of TensorFlow padding 'SAME' for MaxPool2d
            # 第二个零填充层，用于在卷积后保持特征图尺寸。
            nn.MaxPool2d(3, stride=2, padding=0),
            # 第二个最大池化层，使用 3x3 的池化窗口，步长为 2，填充为 0。
        )

        self.classifier = nn.Sequential(
            # 初始化一个 nn.Sequential 容器，用于分类器，包含全连接层。
            nn.Flatten(),
            # Flatten 层，将多维输入张量展平为一维。
            nn.Linear(64*6*6, 384),
            # 第一个全连接层，将展平后的输入特征从 6466 维映射到 384 维。这里的 6*6 来自两次池化后的特征图尺寸。
            nn.ReLU(),
            # 第三个 ReLU 激活函数层。
            nn.Linear(384, 192),
            # 第二个全连接层，将 384 维特征映射到 192 维。
            nn.ReLU(),
            # 第四个 ReLU 激活函数层。
            nn.Linear(192, num_classes),
            # 第三个全连接层，将 192 维特征映射到 num_classes 维，对应于输出类别的数量。
        )

    def forward(self, x):
        # 定义 forward 方法，用于指定模型的前向传播过程。
        x = self.resize(x)
        # 将输入 x 通过 Resize 变换调整尺寸。
        x = self.feature_extractor(x)
        # 将调整尺寸后的输入 x 通过特征提取器进行卷积、池化和归一化操作。
        x = self.classifier(x)
        # 将特征提取器的输出通过分类器进行全连接层操作。
        return x


# From "Gradient-Based Learning Applied to Document Recognition"
class lenet5_orig(nn.Module):
    # 定义了一个名为 lenet5_orig 的类，用于构建 LeNet-5 原始模型。
    def __init__(self, num_classes, num_channels, model_args):
        # 构造函数接收三个参数：
        # num_classes：输出类别的数量。
        # num_channels：输入图像的通道数。
        # model_args：模型参数的字典，可能包含其他用于初始化模型的参数。
        super(lenet5_orig, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        orig_activation = True
        # 标志变量，用于决定是否使用原始 LeNet-5 模型中的激活函数（Tanh）。
        orig_norm = True
        # 标志变量，用于决定是否使用原始 LeNet-5 模型中的批量归一化处理。
        orig_s = True
        # 标志变量，用于决定是否使用原始 LeNet-5 模型中的池化层。
        orig_c3 = True
        # 标志变量，用于决定是否使用原始 LeNet-5 模型中的第三个卷积层。
        orig_f7 = True
        # 标志变量，用于决定是否使用原始 LeNet-5 模型中的全连接层。

        activation = nn.Tanh if orig_activation else nn.ReLU
        # 根据 orig_activation 的值，选择使用 nn.Tanh 或 nn.ReLU 作为激活函数。
        activation_constant = 1.7159 if orig_activation else 1
        # 根据 orig_activation 的值，选择激活函数的常数。
        norm = nn.BatchNorm2d if orig_norm else nn.Identity
        # 根据 orig_norm 的值，选择使用 nn.BatchNorm2d 或 nn.Identity（恒等变换）。
        c1 = nn.Conv2d(num_channels, 6, 5)
        # 第一个卷积层，将输入通道映射到 6 个输出通道，使用 5x5 的卷积核。
        s2 = LeNet5_Orig_S(6) if orig_s else nn.MaxPool2d(2, 2)
        # 根据 orig_s 的值，选择使用自定义的 LeNet5_Orig_S 池化层或标准的 nn.MaxPool2d。
        c3 = LeNet5_Orig_C3() if orig_c3 else nn.Conv2d(6, 16, 5)
        # 根据 orig_c3 的值，选择使用自定义的 LeNet5_Orig_C3 卷积层或标准的 nn.Conv2d。
        s4 = LeNet5_Orig_S(16) if orig_s else nn.MaxPool2d(2, 2)
        # 根据 orig_s 的值，选择使用自定义的 LeNet5_Orig_S 池化层或标准的 nn.MaxPool2d。
        c5 = nn.Conv2d(16, 120, 5, bias=True)
        # 第二个卷积层，将 16 个输入通道映射到 120 个输出通道，使用 5x5 的卷积核。
        f6 = nn.Linear(120, 84)
        # 第一个全连接层，将 120 维特征映射到 84 维。
        f7 = LeNet5_Orig_F7(84, 10) if orig_f7 else nn.Linear(84, 10)
        # 根据 orig_f7 的值，选择使用自定义的 LeNet5_Orig_F7 全连接层或标准的 nn.Linear。
        self.resize = Resize((32, 32))
        # 初始化一个 Resize 变换，将输入图像调整为 32x32 像素。
        self.feature_extractor = nn.Sequential(
            # 初始化一个 nn.Sequential 容器，用于特征提取，包含多个按顺序执行的层。
            c1,
            norm(6),
            activation(), Multiply(activation_constant),
            s2,
            c3,
            norm(16),
            activation(), Multiply(activation_constant),
            s4,
            c5,
            norm(120),
            activation(), Multiply(activation_constant),
        )
        # 添加卷积层、归一化层、激活函数层和池化层到特征提取器中。
        self.classifier = nn.Sequential(
            # 初始化一个 nn.Sequential 容器，用于分类器，包含全连接层。
            nn.Flatten(),
            f6,
            activation(), Multiply(activation_constant),
            f7,
        )
        # 添加 Flatten 层、全连接层和激活函数层到分类器中。
    def forward(self, x):
        # 定义 forward 方法，用于指定模型的前向传播过程。
        x = self.resize(x)
        # 将输入 x 通过 Resize 变换调整尺寸。
        x = self.feature_extractor(x)
        # 将调整尺寸后的输入 x 通过特征提取器进行卷积、归一化、激活和池化操作。
        x = self.classifier(x)
        # 将特征提取器的输出通过分类器进行全连接层操作。
        return x

# From "Communication-Efficient Learning of Deep Networks from Decentralized Data"
# (ported from 2016 TensorFlow CIFAR-10 tutorial)
#     * LocalResponseNorm replaced with BatchNorm2d/GroupNorm/Identity
#     * Normalization placed always before ReLU
#     * Conv2d-Normalization-ReLU optionally replaced by GhostModule from
#     "GhostNet: More Features from Cheap Operations"


class lenet5(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        # 构造函数接收三个参数：
        # num_classes：输出类别的数量。
        # num_channels：输入图像的通道数。
        # model_args：模型参数的字典，可能包含其他用于初始化模型的参数。
        super(lenet5, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        norm = model_args['norm'] if 'norm' in model_args else 'batch'
        # 从 model_args 中获取归一化类型，如果未提供，则默认为 'batch'。
        if norm == 'batch':
            norm1 = nn.BatchNorm2d(64)
            norm2 = nn.BatchNorm2d(64)
        elif norm == 'group':
            # Group Normalization paper suggests 16 channels per group is best
            norm1 = nn.GroupNorm(int(64/16), 64)
            norm2 = nn.GroupNorm(int(64/16), 64)
        elif norm == None:
            norm1 = nn.Identity(64)
            norm2 = nn.Identity(64)
        else:
            raise ValueError("Unsupported norm '%s' for LeNet5")
        # 根据 norm 的值，初始化两个归一化层 norm1 和 norm2。支持批量归一化（BatchNorm2d）、组归一化（GroupNorm）和恒等变换（Identity）。
        if 'ghost' in model_args and model_args['ghost']:
            block1 = GhostModule(num_channels, 64, 5, padding='same', norm=norm)
            block2 = GhostModule(64, 64, 5, padding='same', norm=norm)
        else:
            block1 = nn.Sequential(
                nn.Conv2d(num_channels, 64, 5, padding='same'),
                norm1,
                nn.ReLU(),
            )
            block2 = nn.Sequential(
                nn.Conv2d(64, 64, 5, padding='same'),
                norm2,
                nn.ReLU(),
            )
        # 根据 model_args 中是否包含 'ghost' 参数，决定是否使用 GhostModule。
        # 如果没有使用 GhostModule，则使用标准的卷积层、归一化层和 ReLU 激活函数。
        self.resize = Resize((24, 24))
        # 初始化一个 Resize 变换，将输入图像调整为 24x24 像素。
        self.feature_extractor = nn.Sequential(
            # 初始化一个 nn.Sequential 容器，用于特征提取，包含多个按顺序执行的层。
            block1,
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(3, stride=2, padding=0),
            block2,
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(3, stride=2, padding=0),
        )
        # 添加卷积层、零填充层和最大池化层到特征提取器中。
        self.classifier = nn.Sequential(
            # 初始化一个 nn.Sequential 容器，用于分类器，包含全连接层。
            nn.Flatten(),
            nn.Linear(64*6*6, 384), # 5*5 if input is 32x32
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, num_classes))
            # 添加 Flatten 层、全连接层和 ReLU 激活函数层到分类器中

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class mnasnet(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        # 构造函数接收三个参数：
        # num_classes：输出类别的数量。
        # num_channels：输入图像的通道数。
        # model_args：模型参数的字典，可能包含其他用于初始化模型的参数。
        super(mnasnet, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        width = model_args['width'] if 'width' in model_args else 1
        # 从 model_args 中获取模型宽度因子，如果未提供，则默认为 1。
        dropout = model_args['dropout'] if 'dropout' in model_args else 0.2
        # 从 model_args 中获取 dropout 比率，如果未提供，则默认为 0.2。
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        # 从 model_args 中获取是否使用预训练模型的标记，如果未提供，则默认为 False。
        freeze = model_args['freeze'] if 'freeze' in model_args else False
        # 从 model_args 中获取是否冻结模型权重的标记，如果未提供，则默认为 False。
        self.resize = Resize(224)
        # 初始化一个 Resize 变换，将输入图像调整为 224x224 像素，这是 MNASNet 预训练模型通常使用的输入尺寸。
        if pretrained:
            # 如果使用预训练模型：
            if width == 1:
                self.model = tvmodels.mnasnet1_0(pretrained=True, dropout=dropout)
            elif width == 0.5:
                self.model = tvmodels.mnasnet0_5(pretrained=True, dropout=dropout)
            elif width == 0.75:
                self.model = tvmodels.mnasnet0_75(pretrained=True, dropout=dropout)
            elif width == 1.3:
                self.model = tvmodels.mnasnet1_3(pretrained=True, dropout=dropout)
            else:
                raise ValueError('Unsupported width for pretrained MNASNet: %s' % width)
            # 根据 width 的值，选择相应的预训练 MNASNet 模型。每个模型有不同的宽度因子，影响模型的复杂度和性能。
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            #  如果冻结模型权重，则将模型的所有参数的 requires_grad 属性设置为 False，这样在训练过程中这些参数不会更新。
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
            # 替换模型的分类器层，以匹配输出类别的数量。
        else:
            self.model = tvmodels.mnasnet.MNASNet(alpha=width, num_classes=num_classes, dropout=dropout)
            # 如果不使用预训练模型，则直接初始化一个 MNASNet 模型，指定宽度因子、输出类别数量和 dropout 比率。

    def forward(self, x):
        # 定义 forward 方法，用于指定模型的前向传播过程。
        x = self.resize(x)
        # 将输入 x 通过 Resize 变换调整尺寸。
        x = self.model(x)
        # 将调整尺寸后的输入 x 通过模型进行前向传播。
        return x
        # 返回模型的输出，即分类的 logits 或概率。


class ghostnet(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        # 构造函数接收三个参数：
        # num_classes：输出类别的数量。
        # num_channels：输入图像的通道数。
        # model_args：模型参数的字典，可能包含其他用于初始化模型的参数。
        super(ghostnet, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        width = model_args['width'] if 'width' in model_args else 1.0
        # 从 model_args 中获取模型宽度因子，如果未提供，则默认为 1.0。
        dropout = model_args['dropout'] if 'dropout' in model_args else 0.2
        # 从 model_args 中获取 dropout 比率，如果未提供，则默认为 0.2。
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        # 从 model_args 中获取是否使用预训练模型的标记，如果未提供，则默认为 False。
        freeze = model_args['freeze'] if 'freeze' in model_args else False
        # 从 model_args 中获取是否冻结模型权重的标记，如果未提供，则默认为 False。
        self.resize = Resize(224)
        #self.resize = Resize(24)
        # 初始化一个 Resize 变换，将输入图像调整为 224x224 像素，这是 GhostNet 预训练模型通常使用的输入尺寸。
        if pretrained:
            # 如果使用预训练模型：
            if width != 1:
                raise ValueError('Unsupported width for pretrained GhostNet: %s' % width)
            # 检查模型宽度因子是否为 1，因为预训练的 GhostNet 模型通常只有一种宽度因子。
            self.model = load_ghostnet(width=1, dropout=dropout)
            # 加载预训练的 GhostNet 模型。
            self.model.load_state_dict(torch.load('models/ghostnet.pth'), strict=True)
            # 从文件中加载 GhostNet 模型的状态字典，并使用严格模式进行加载。
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            # 如果冻结模型权重，则将模型的所有参数的 requires_grad 属性设置为 False，这样在训练过程中这些参数不会更新。
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
            # 替换模型的分类器层，以匹配输出类别的数量。
        else:
            self.model = load_ghostnet(num_classes=num_classes, width=width, dropout=dropout)
            # 如果不使用预训练模型，则直接初始化一个 GhostNet 模型，指定输出类别数量、模型宽度因子和 dropout 比率。

    def forward(self, x):
        # 定义 forward 方法，用于指定模型的前向传播过程。
        x = self.resize(x)
        # 将输入 x 通过 Resize 变换调整尺寸。
        x = self.model(x)
        # 将调整尺寸后的输入 x 通过模型进行前向传播。
        return x
        # 返回模型的输出，即分类的 logits 或概率。

'''
class tinynet(nn.Module):
    variants = {'a': (0.86, 1.0, 1.2),
                'b': (0.84, 0.75, 1.1),
                'c': (0.825, 0.54, 0.85),
                'd': (0.68, 0.54, 0.695),
                'e': (0.475, 0.51, 0.60)}

    def __init__(self, num_classes, num_channels, model_args):
        super(tinynet, self).__init__()

        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        freeze = model_args['freeze'] if 'freeze' in model_args else False
        r = model_args['r'] if 'r' in model_args else tinynet.variants['a'][0]
        w = model_args['w'] if 'w' in model_args else tinynet.variants['a'][1]
        d = model_args['d'] if 'd' in model_args else tinynet.variants['a'][2]
        if 'variant' in model_args:
            variant = model_args['variant']
            if variant not in tinynet.variants:
                raise ValueError(f'Non existent variant for TinyNet: {variant}')
            r, w, d = tinynet.variants[variant]
        else:
            variant = None
            for key in tinynet.variants:
                if (r, w, d) == tinynet.variants[key]:
                    variant = key
                    break

        self.resize = Resize(round(224*r))

        self.model = load_tinynet(r=r, w=w, d=d)
        if pretrained:
            if variant is None:
                raise ValueError(f'Unsupported r, w, d for pretrained TinyNet: {r}, {w}, {d}')
            self.model.load_state_dict(torch.load(f'models/tinynet_{variant}.pth'), strict=True)
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x
'''


class mobilenet_v3(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        # 构造函数接收三个参数：
        # num_classes：输出类别的数量。
        # num_channels：输入图像的通道数。
        # model_args：模型参数的字典，可能包含其他用于初始化模型的参数。
        super(mobilenet_v3, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        variant = model_args['variant'] if 'variant' in model_args else 'small'
        # 从 model_args 中获取模型变体，如果未提供，则默认为 'small'。MobileNetV3 有 'small' 和 'large' 两种变体。
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        # 从 model_args 中获取是否使用预训练模型的标记，如果未提供，则默认为 False。
        freeze = model_args['freeze'] if 'freeze' in model_args else False
        # 从 model_args 中获取是否冻结模型权重的标记，如果未提供，则默认为 False。
        self.resize = Resize(224)
        # 初始化一个 Resize 变换，将输入图像调整为 224x224 像素，这是 MobileNetV3 预训练模型通常使用的输入尺寸。
        self.model = getattr(tvmodels, f'mobilenet_v3_{variant}')(pretrained=pretrained)
        # 根据模型变体（'small' 或 'large'）动态获取相应的 MobileNetV3 模型，并设置为预训练模式。
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        # 如果使用预训练模型且需要冻结权重，则将模型的所有参数的 requires_grad 属性设置为 False，这样在训练过程中这些参数不会更新。
        self.model.classifier[0] = nn.Linear(self.model.classifier[0].in_features, self.model.classifier[0].out_features)
        # 重新初始化模型的第一个分类器层，保持其输入和输出特征数量不变。
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
        # 替换模型的最后一个分类器层，将输出特征数量更改为输出类别的数量 num_classes。
    def forward(self, x):
        x = self.resize(x)
        # 将输入 x 通过 Resize 变换调整尺寸。
        x = self.model(x)
        # 将调整尺寸后的输入 x 通过模型进行前向传播。
        return x
        # 返回模型的输出，即分类的 logits 或概率。


class efficientnet(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        # 构造函数接收三个参数：
        # num_classes：输出类别的数量。
        # num_channels：输入图像的通道数。
        # model_args：模型参数的字典，可能包含其他用于初始化模型的参数。
        super(efficientnet, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        variant = model_args['variant'] if 'variant' in model_args else 'b0'
        # 从 model_args 中获取 EfficientNet 的变体版本，如果未提供，则默认为 'b0'。EfficientNet 有多个变体，例如 'b0', 'b1', 'b2', 等。
        pretrained = model_args['pretrained'] if 'pretrained' in model_args else False
        # 从 model_args 中获取是否使用预训练模型的标记，如果未提供，则默认为 False。
        freeze = model_args['freeze'] if 'freeze' in model_args else False
        # 从 model_args 中获取是否冻结模型权重的标记，如果未提供，则默认为 False。
        self.resize = Resize(224)
        # 初始化一个 Resize 变换，将输入图像调整为 224x224 像素，这是 EfficientNet 预训练模型通常使用的输入尺寸。
        self.model = getattr(tvmodels, f'efficientnet_{variant}')(pretrained=pretrained)
        # 根据 EfficientNet 的变体版本动态获取相应的 EfficientNet 模型，并设置为预训练模式。
        if pretrained:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
        # 如果使用预训练模型且需要冻结权重，则将模型的所有参数的 requires_grad 属性设置为 False，这样在训练过程中这些参数不会更新。
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        # 替换模型的分类器层，将输出特征数量更改为输出类别的数量 num_classes。

    def forward(self, x):
        x = self.resize(x)
        # 将输入 x 通过 Resize 变换调整尺寸。
        x = self.model(x)
        # 将调整尺寸后的输入 x 通过模型进行前向传播。
        return x
