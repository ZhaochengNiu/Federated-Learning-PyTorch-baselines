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

from math import ceil
# 从Python标准库的math模块导入ceil函数。ceil函数用于执行向上取整操作，即找到一个不小于给定数值的最小整数。
import torch
# 导入PyTorch库，这是一个广泛使用的开源机器学习库，特别适合处理基于GPU的张量计算。
from torch import nn
# 从PyTorch库中导入nn模块，它包含构建神经网络所需的类和函数，例如层、损失函数和优化器。
import torchvision.transforms as tvtransforms
# 导入torchvision.transforms模块，并将其别名设置为tvtransforms。这个模块提供了一系列图像变换操作，用于图像预处理和数据增强。


class GhostModule(nn.Module):
    # 定义了一个名为 GhostModule 的类，用于构建 Ghost 模块，它继承自 PyTorch 的 nn.Module 类。
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, padding=0, relu=True, norm='batch'):
        # 构造函数接收多个参数，用于初始化 Ghost 模块：
        # inp：输入通道数。
        # oup：输出通道数。
        # kernel_size：卷积核大小，默认为 1。
        # ratio：Ghost 模块的比率，用于计算额外的通道数。
        # dw_size：深度可分离卷积的卷积核大小。
        # stride：步长，默认为 1。
        # padding：填充，默认为 0。
        # relu：是否使用 ReLU 激活函数。
        # norm：归一化类型，可以是 'batch'、'group' 或 None。
        super(GhostModule, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        self.oup = oup
        # 保存输出通道数。
        init_channels = ceil(oup / ratio)
        # 计算初始通道数，使用 math.ceil 向上取整。
        new_channels = init_channels*(ratio-1)
        # 根据比率计算额外的通道数。
        if norm == 'batch':
            # 根据归一化类型判断，如果使用批量归一化：
            norm1 = nn.BatchNorm2d(init_channels)
            norm2 = nn.BatchNorm2d(new_channels)
            # 初始化批量归一化层。
        elif norm == 'group':
            # 如果使用组归一化：
            norm1 = nn.GroupNorm(int(init_channels/16), init_channels)
            norm2 = nn.GroupNorm(int(new_channels/16), new_channels)
            # 初始化组归一化层。
        elif norm == None:
            # 如果不需要归一化：
            norm1 = nn.Identity(init_channels)
            norm2 = nn.Identity(new_channels)
            # 使用恒等变换作为归一化层。
        self.primary_conv = nn.Sequential(
            # 定义主要卷积操作的序列：
            nn.Conv2d(inp, init_channels, kernel_size, stride, padding, bias=False),
            norm1,
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        # 包括卷积层、归一化层和可选的 ReLU 激活函数。
        self.cheap_operation = nn.Sequential(
            # 定义廉价操作的序列，用于生成额外的通道：
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            norm2,
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        # 包括深度可分离卷积、归一化层和可选的 ReLU 激活函数。

    def forward(self, x):
        # 定义 forward 方法，用于前向传播。
        x1 = self.primary_conv(x)
        # 将输入 x 通过主要卷积操作。
        x2 = self.cheap_operation(x1)
        # 将 x1 的结果通过廉价操作。
        out = torch.cat([x1,x2], dim=1)
        # 沿着通道维度拼接 x1 和 x2 的结果。
        return out[:,:self.oup,:,:]
        # 返回拼接结果的前 self.oup 个通道，确保输出通道数符合预期。


class LeNet5_Orig_S(nn.Module):
    # 这段代码定义了一个名为 LeNet5_Orig_S 的类，它是一个 PyTorch nn.Module 的子类，用于模拟原始 LeNet-5 网络中的池化操作。
    # 以下是对类及其方法的逐行解释：
    def __init__(self, in_channels=None):
        # 定义了一个名为 LeNet5_Orig_S 的类，它继承自 PyTorch 的 nn.Module 类，用于表示神经网络的模块。
        super(LeNet5_Orig_S, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        # The number of parameters is 2 * in_channels. 这是一条注释，说明接下来的卷积层将有 2 * in_channels 个参数。
        self.s1 = nn.AvgPool2d(2, 2)
        # 定义了一个平均池化层 s1，使用 2x2 的池化窗口和步长，这将减少特征图的尺寸。
        self.s2 = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=True)
        # 定义了一个深度为 1 的卷积层 s2，使用 in_channels 作为输入和输出通道数，groups=in_channels 表示这是一个深度可分离卷积，
        # bias=True 表示卷积层有偏置项。这个卷积层用于在池化后调整特征通道。

    def forward(self, x):
        # 定义了 forward 方法，用于执行前向传播。这个方法接收一个参数 x，它是输入的张量。
        x = self.s1(x)
        # 将输入 x 通过平均池化层 s1。
        x = self.s2(x)
        # 将池化后的结果 x 通过卷积层 s2。
        return x
        # 返回经过池化和卷积处理后的输出张量 x。


class LeNet5_Orig_C3(nn.Module):
    # 定义了一个名为 LeNet5_Orig_C3 的类，它继承自 PyTorch 的 nn.Module 类。
    """The original C3 conv. layer as described in "Gradient-Based Learning Applied
    to Document Recognition", by LeCun et al.
    """
    # 这是一个多行字符串（文档字符串），用于描述这个类实现的是 LeNet-5 网络中的 C3 卷积层，
    # 这个层的描述来源于 LeCun 等人的论文 "Gradient-Based Learning Applied to Document Recognition"。
    def __init__(self):
        # 构造函数，没有接收任何参数。
        super(LeNet5_Orig_C3, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        # The connections are shown in Table 1 in the paper.
        self.s2_ch_3_in = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5]
        ]
        # 定义一个列表 s2_ch_3_in，包含从 S2 层的 3 个输入通道到 C3 层的索引映射。
        self.s2_ch_4_in = [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 1, 2, 5],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5]
        ]
        # 定义一个列表 s2_ch_4_in，包含从 S2 层的 4 个输入通道到 C3 层的索引映射。
        # The number of parameters is 6 * (75 + 1) + 9 * (100 + 1) + 1 * (150 + 1) = 1516.
        self.c3_3_in = nn.ModuleList()
        # 创建一个 ModuleList，用于存储从 3 个输入通道到 C3 层的卷积模块。
        self.c3_4_in = nn.ModuleList()
        # 创建一个 ModuleList，用于存储从 4 个输入通道到 C3 层的卷积模块。
        for i in range(6):
            self.c3_3_in.append(nn.Conv2d(3, 1, 5, padding=0))
        # 为 s2_ch_3_in 定义的每个索引映射创建一个卷积层，使用 5x5 的卷积核和 0 的填充。
        for i in range(9):
            self.c3_4_in.append(nn.Conv2d(4, 1, 5, padding=0))
        # 为 s2_ch_4_in 定义的每个索引映射创建一个卷积层。
        self.c3_6_in = nn.Conv2d(6, 1, 5, padding=0)
        # 创建一个卷积层，用于处理所有 6 个输入通道。

    def forward(self, x):
        # 定义 forward 方法，用于执行前向传播。这个方法接收一个参数 x，它是输入的张量。
        c3 = []
        # 初始化一个列表 c3，用于存储 C3 层的输出。
        for i in range(6):
            c3.append(self.c3_3_in[i](x[:, self.s2_ch_3_in[i], :, :]))
        # 对于每个从 3 个输入通道到 C3 层的卷积模块，提取输入张量 x 的相应通道，并通过卷积模块。
        for i in range(9):
            c3.append(self.c3_4_in[i](x[:, self.s2_ch_4_in[i], :, :]))
        # 对于每个从 4 个输入通道到 C3 层的卷积模块，执行相同的操作。
        c3.append(self.c3_6_in(x))
        # 将所有 6 个输入通道通过最后一个卷积层。
        x = torch.cat(c3, dim=1)
        # 沿着通道维度（dim=1）拼接所有 C3 层的输出。
        return x
        # 返回拼接后的输出张量 x。


class LeNet5_Orig_F7(nn.Module):
    # 这段代码定义了一个名为 LeNet5_Orig_F7 的类，它是 PyTorch nn.Module 的子类，
    # 用于模拟原始 LeNet-5 网络中的 F7 层，该层是一个基于欧几里得距离的相似度度量层。以下是对类及其方法的逐行解释：
    def __init__(self, in_features, out_features):
        # 构造函数接收两个参数：
        # in_features：输入特征的数量。
        # out_features：输出特征（或类别）的数量。
        super(LeNet5_Orig_F7, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        self.in_features = in_features
        self.out_features = out_features
        # 保存输入和输出特征的数量。
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        # 创建一个可学习的参数 centers，它是一个张量，其形状为 (out_features, in_features)，表示每个类别的中心点。
        nn.init.normal_(self.centers, 0, 1)
        # 使用标准正态分布初始化 centers 参数，均值为 0，标准差为 1。

    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        # 计算输出张量的大小，它将具有与输入相同的批次大小，以及 out_features 和 in_features 的维度。
        x = x.unsqueeze(1).expand(size)
        # 将输入张量 x 增加一个维度，并扩展其大小以匹配 size。这将 x 转换为一个三维张量，
        # 其中第一个维度是批次大小，第二个维度是输出特征数量，第三个维度是输入特征数量。
        c = self.centers.unsqueeze(0).expand(size)
        # 将 centers 参数增加一个批次维度，并扩展其大小以匹配 size。这将 centers 转换为与 x 相同的三维张量。
        return (x - c).pow(2).sum(-1)
        # 计算输入 x 和中心点 c 之间的平方欧几里得距离，通过对差的平方求和来实现。
        # 使用 pow(2) 计算平方，然后使用 sum(-1) 沿着最后一个维度（输入特征维度）求和，得到每个批次和每个类别的相似度分数。


class Multiply(nn.Module):
    # 定义了一个名为 Multiply 的类，它继承自 PyTorch 的 nn.Module 类。
    def __init__(self, k):
        # 构造函数接收一个参数 k，这个参数将用于后续的乘法操作。
        super(Multiply, self).__init__()
        # 这样确保了父类 nn.Module 的构造函数被正确调用。
        self.k = k
        # 保存传入的参数 k 作为实例变量。

    def forward(self, x):
        return x*self.k
        # 执行乘法操作，将输入张量 x 与实例变量 self.k 相乘，然后返回结果。

'''
def conv_out_size(s_in, kernel_size, padding, stride):
    if padding == 'same':
        s_out = (ceil(s_in[0]/stride[0]), ceil(s_in[1]/stride[1]))
        padding_h = max((s_out[0] - 1)*stride[0] + kernel_size[0] - s_in[0], 0)
        padding_w = max((s_out[1] - 1)*stride[1] + kernel_size[1] - s_in[1], 0)
        padding_l = padding_w//2
        padding_r = padding_w - padding_l
        padding_t = padding_h//2
        padding_b = padding_h - padding_t
        return s_out, (padding_l, padding_r, padding_t, padding_b)

    h_out = int((s_in[0] - kernel_size[0] + padding[2] + padding[3])/stride[0] + 1)
    w_out = int((s_in[1] - kernel_size[1] + padding[0] + padding[1])/stride[1] + 1)

    return (h_out, w_out), padding
'''
'''
def create_combined_model(model_fe):

    num_ftrs = model_fe.fc.in_features

    model_fe_features = nn.Sequential(
        model_fe.quant,  # Quantize the input
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.layer4,
        model_fe.avgpool,
        model_fe.dequant,  # Dequantize the output
    )

    # Step 2. Create a new "head"
    new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2),
    )

    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
    )
    return new_model
'''
