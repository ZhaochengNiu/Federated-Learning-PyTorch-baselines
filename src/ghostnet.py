# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
# 导入PyTorch库，这是一个广泛使用的开源机器学习库，特别适合处理基于GPU的张量计算。
import torch.nn as nn
# 从PyTorch库中导入torch.nn模块，它包含构建神经网络所需的类和函数。
import torch.nn.functional as F
# 从PyTorch库中导入torch.nn.functional模块，它包含了一系列函数式接口的神经网络层和操作。
import math
# 导入Python的math模块，提供了一系列数学函数。


__all__ = ['ghost_net']
# 定义了一个特殊的变量__all__，它是一个字符串列表，用于指定当其他模块使用from <module> import *语句时应该导入的名称。
# 在这个例子中，它指定了ghost_net应该是使用from <module> import *时唯一导入的名称。


def _make_divisible(v, divisor, min_value=None):
    # 这段代码定义了一个名为 _make_divisible 的函数，它用于确保某个数值可以被给定的除数整除，通常用于确定网络层中的通道数。
    # 以下是对函数及其参数的逐行解释：
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    # 定义了一个名为 _make_divisible 的函数，它接收三个参数：
    # v：要调整的数值。
    # divisor：数值需要能被整除的除数。
    # min_value：可选参数，数值的最小值，默认为 None。
    if min_value is None:
        min_value = divisor
    # 如果 min_value 参数没有提供（即为 None），则将其设置为 divisor 的值。
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 计算 v 的新值 new_v，使其能被 divisor 整除。首先，将 v 加上 divisor / 2 以保证四舍五入时的准确性，然后向下取整（// 运算符），
    # 最后乘以 divisor 确保结果可以被整除。同时，使用 max 函数确保 new_v 不小于 min_value。
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        # 这一行是注释，说明接下来的代码用于确保向下取整操作不会使结果小于原数值的 90%。
        new_v += divisor
        # 如果新计算的 new_v 小于原数值 v 的 90%，则将 new_v 增加 divisor 的值，以确保不会减少太多。
    return new_v
    # 返回调整后的 new_v 值。


# 硬饱和激活函数（Hard Sigmoid）是一种近似于 Sigmoid 激活函数的函数，但在实现上更高效，
# 因为它使用线性函数和 ReLU 变体来近似 Sigmoid 的非线性特性。这个函数可以用就地操作来实现，以减少内存消耗，
# 或者使用标准的非就地操作。在 PyTorch 中，add_、clamp_ 和 div_ 方法用于就地操作，
# 而 F.relu6 是一个函数式接口，用于应用 ReLU6 激活函数。
def hard_sigmoid(x, inplace: bool = False):
    # 定义了一个名为 hard_sigmoid 的函数，它接收两个参数：
    # x：输入的张量。
    # inplace：一个布尔值，默认为 False，表示是否使用就地（in-place）操作来减少内存使用。
    if inplace:
        # 如果 inplace 参数为 True，则执行以下就地操作。
        return x.add_(3.).clamp_(0., 6.).div_(6.)
        # 执行以下操作：
        # x.add_(3.)：将输入张量 x 每个元素加上 3，使用就地加法。
        # clamp_(0., 6.)：将上一步的结果限制在 0 到 6 之间，使用就地操作。
        # div_(6.)：将限制后的结果除以 6，使用就地除法。
        # 然后返回处理后的张量。
    else:
        # 如果 inplace 参数为 False，则执行以下操作。
        return F.relu6(x + 3.) / 6.
        # 执行以下操作：
        # x + 3.：输入张量 x 每个元素加上 3。
        # F.relu6()：应用 ReLU6 激活函数，它将输入张量中的每个元素限制在 0 到 6 之间。
        #  / 6.：将 ReLU6 激活函数的结果除以 6。
        # 然后返回处理后的张量。


# SqueezeExcite 类实现了 Squeeze-and-Excitation 模块，通过在卷积层后添加一个通道注意力机制来提高网络的表征能力。
# 这种机制通过全局平均池化获取空间信息，然后通过两层卷积层分别缩减和扩展通道数，最后使用门控函数对原始特征图的通道进行加权。
class SqueezeExcite(nn.Module):
    # 这段代码定义了一个名为 SqueezeExcite 的类，它实现了 Squeeze-and-Excitation (SE) 模块，这是一种通道注意力机制，
    # 用于提高卷积神经网络的性能。以下是对类及其方法的逐行解释：
    # 定义了一个名为 SqueezeExcite 的类，它继承自 PyTorch 的 nn.Module 类。
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        # 构造函数接收多个参数：
        # in_chs：输入特征图的通道数。
        # se_ratio：SE模块中缩减通道的比例，默认为0.25。
        # reduced_base_chs：可选参数，用于计算缩减后的通道数的基础通道数。
        # act_layer：激活层，默认使用 nn.ReLU。
        # gate_fn：门控函数，默认使用 hard_sigmoid。
        # divisor：用于调整通道数的除数，默认为4。
        # _：捕获其他关键字参数。
        super(SqueezeExcite, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        self.gate_fn = gate_fn
        # 保存门控函数。
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        # 计算缩减后的通道数，使用 _make_divisible 函数确保结果是可整除的。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 创建一个自适应平均池化层，用于将特征图的大小缩减到 (1, 1)。
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        # 创建一个卷积层，用于将输入通道缩减到 reduced_chs。
        self.act1 = act_layer(inplace=True)
        # 创建激活层，使用就地操作。
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        # 创建另一个卷积层，用于将缩减后的通道数扩展回原始的 in_chs。

    def forward(self, x):
        x_se = self.avg_pool(x)
        # 通过自适应平均池化获取全局空间信息。
        x_se = self.conv_reduce(x_se)
        # 通过卷积层缩减通道。
        x_se = self.act1(x_se)
        # 应用激活函数。
        x_se = self.conv_expand(x_se)
        # 通过卷积层扩展通道回原始数量。
        x = x * self.gate_fn(x_se)
        # 将门控函数应用于缩减并扩展后的张量 x_se，然后与原始特征图 x 相乘，实现通道注意力机制。
        return x
        # 返回处理后的张量。


class ConvBnAct(nn.Module):
    # 这段代码定义了一个名为 ConvBnAct 的类，它是一个 PyTorch nn.Module 的子类，
    # 用于实现一个包含卷积、批量归一化和激活函数的神经网络层。以下是对类及其方法的逐行解释：
    # 定义了一个名为 ConvBnAct 的类，它继承自 PyTorch 的 nn.Module 类。
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        # 构造函数接收以下参数：
        # in_chs：输入特征图的通道数。
        # out_chs：输出特征图的通道数。
        # kernel_size：卷积核的大小。
        # stride：卷积的步长，默认为 1。
        # act_layer：激活层，默认使用 nn.ReLU。
        super(ConvBnAct, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        # 创建一个卷积层，参数包括：
        # in_chs：输入通道数。
        # out_chs：输出通道数。
        # kernel_size：卷积核大小。
        # stride：步长。
        # kernel_size//2：填充大小，确保输出特征图大小与输入特征图相同。
        # bias=False：卷积层不使用偏置项。
        self.bn1 = nn.BatchNorm2d(out_chs)
        # 创建一个批量归一化层，参数为 out_chs，即卷积层的输出通道数。
        self.act1 = act_layer(inplace=True)
        # 创建一个激活层，使用就地操作（如果可能）。

    def forward(self, x):
        x = self.conv(x)
        # 将输入 x 通过卷积层。
        x = self.bn1(x)
        # 将卷积层的输出通过批量归一化层。
        x = self.act1(x)
        # 将批量归一化层的输出通过激活层。
        return x
        # 返回处理后的张量。


class GhostModule(nn.Module):
    # 这段代码定义了一个名为 GhostModule 的类，它是一个 PyTorch nn.Module 的子类，
    # 用于实现 GhostNet 架构中的 Ghost 模块。以下是对类及其方法的逐行解释：
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        # 构造函数接收以下参数：
        # inp：输入特征图的通道数。
        # oup：输出特征图的通道数。
        # kernel_size：卷积核的大小，默认为 1。
        # ratio：Ghost 模块的比率，用于计算额外的通道数。
        # dw_size：深度可分离卷积的卷积核大小，默认为 3。
        # stride：卷积的步长，默认为 1。
        # relu：一个布尔值，指示是否使用 ReLU 激活函数。
        super(GhostModule, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        self.oup = oup
        # 保存输出通道数。
        init_channels = math.ceil(oup / ratio)
        # 计算初始通道数，使用 math.ceil 向上取整。
        new_channels = init_channels*(ratio-1)
        # 根据比率计算额外的通道数。
        self.primary_conv = nn.Sequential(
            # 定义主要卷积操作的序列：
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            #  创建一个卷积层，将输入通道映射到初始通道。
            nn.BatchNorm2d(init_channels),
            # 添加批量归一化层。
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            # 如果 relu 为 True，则添加就地 ReLU 激活函数。
        )
        self.cheap_operation = nn.Sequential(
            # 定义廉价操作的序列，用于生成额外的通道：
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            # 创建一个深度可分离卷积层，使用 init_channels 作为组数。
            nn.BatchNorm2d(new_channels),
            # 添加批量归一化层。
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            # 如果 relu 为 True，则添加就地 ReLU 激活函数。
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        # 将输入 x 通过主要卷积操作。
        x2 = self.cheap_operation(x1)
        # 将主要卷积操作的结果 x1 通过廉价操作。
        out = torch.cat([x1,x2], dim=1)
        # 沿着通道维度拼接 x1 和 x2 的结果。
        return out[:,:self.oup,:,:]
        # 返回拼接结果的前 self.oup 个通道，确保输出通道数符合预期。


# GhostBottleneck 类实现了 GhostNet 架构中的 Ghost 瓶颈层，通过点卷积和深度可分离卷积实现特征的通道变换，
# 并可选地使用 SE 模块进一步提取特征。这种设计允许网络在保持性能的同时减少计算量。
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""
    # 这段代码定义了一个名为 GhostBottleneck 的类，它是一个 PyTorch nn.Module 的子类，用于实现 GhostNet 架构中的 Ghost 瓶颈层。
    # 以下是对类及其方法的逐行解释：
    # 定义了一个名为 GhostBottleneck 的类，它继承自 PyTorch 的 nn.Module 类。
    # 这个类实现了带有可选 Squeeze-and-Excitation (SE) 模块的 Ghost 瓶颈层。
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        # 构造函数接收以下参数：
        # in_chs：输入特征图的通道数。
        # mid_chs：中间特征图的通道数。
        # out_chs：输出特征图的通道数。
        # dw_kernel_size：深度可分离卷积的卷积核大小，默认为 3。
        # stride：卷积的步长，默认为 1。
        # act_layer：激活层，默认使用 nn.ReLU。
        # se_ratio：SE 模块的比率，默认为 0，表示不使用 SE 模块。
        super(GhostBottleneck, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        has_se = se_ratio is not None and se_ratio > 0.
        # 判断是否需要使用 SE 模块。
        self.stride = stride
        # 保存步长。
        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        # 创建第一个 Ghost 模块，用于将输入通道扩展到中间通道。
        # Depth-wise convolution
        if self.stride > 1:
            # 如果步长大于 1，执行深度可分离卷积。
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            # 创建深度可分离卷积层。
            self.bn_dw = nn.BatchNorm2d(mid_chs)
            # 创建批量归一化层。
        # Squeeze-and-excitation
        if has_se:
            # 如果需要使用 SE 模块。
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
            # 创建 SE 模块。
        else:
            # 如果不使用 SE 模块，则将 SE 模块设置为 None。
            self.se = None
        # Point-wise linear projection
        # 创建第二个 Ghost 模块，用于将中间通道映射到输出通道。
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            # 如果输入和输出通道相同，且步长为 1。
            self.shortcut = nn.Sequential()
            # 创建一个空的快捷连接。
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
            # 否则，创建一个包含深度可分离卷积和点卷积的快捷连接。

    def forward(self, x):
        # 定义了 GhostBottleneck 类的 forward 方法，它接收一个参数 x，代表输入的特征图。
        residual = x
        # 保存输入 x 作为残差，用于后续的残差连接。
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # 将输入 x 通过第一个 Ghost 模块 self.ghost1 进行前向传播。
        # Depth-wise convolution
        if self.stride > 1:
            # 检查步长是否大于 1，如果是，则执行深度可分离卷积。
            x = self.conv_dw(x)
            # 将特征图 x 通过深度可分离卷积层 self.conv_dw。
            x = self.bn_dw(x)
            # 将深度可分离卷积的结果通过批量归一化层 self.bn_dw。
        # Squeeze-and-excitation
        if self.se is not None:
            # 检查是否存在 Squeeze-and-Excitation (SE) 模块。
            x = self.se(x)
            # 如果存在 SE 模块，则将特征图 x 通过 SE 模块 self.se。
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        # 将特征图 x 通过第二个 Ghost 模块 self.ghost2 进行前向传播。
        x += self.shortcut(residual)
        # 执行残差连接，将原始输入 residual 与经过 Ghost 模块和可能的深度可分离卷积及 SE 模块处理后的特征图 x 相加。
        return x
        # 返回最终的特征图 x，它将作为下一层的输入。


# GhostNet 类实现了 GhostNet 架构，通过一系列倒置残差块和 Ghost 瓶颈层来提取特征，并通过全局平均池化、
# 最后一个卷积层和分类器来完成图像的分类任务。这种设计使得 GhostNet 在保持计算效率的同时，能够实现优秀的性能。
class GhostNet(nn.Module):
    # 这段代码定义了一个名为 GhostNet 的类，它是一个 PyTorch nn.Module 的子类，用于实现 GhostNet 架构。
    # GhostNet 是一种高效的卷积神经网络，适用于各种图像识别任务。以下是对类及其方法的逐行解释：
    # 定义了一个名为 GhostNet 的类，它继承自 PyTorch 的 nn.Module 类。
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        # 构造函数接收以下参数：
        # cfgs：一个配置列表，定义了网络中每个阶段的 Ghost 瓶颈层的配置。
        # num_classes：输出类别的数量，默认为 1000。
        # width：模型宽度的乘数，默认为 1.0。
        # dropout：Dropout 比率，默认为 0.2。
        super(GhostNet, self).__init__()
        # 调用父类 nn.Module 的构造函数。
        # setting of inverted residual blocks
        self.cfgs = cfgs
        # 保存网络配置。
        self.dropout = dropout
        # 保存 Dropout 比率。
        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        # 计算第一个卷积层的输出通道数。
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        # 创建第一个卷积层 conv_stem。
        self.bn1 = nn.BatchNorm2d(output_channel)
        # 创建第一个批量归一化层 bn1。
        self.act1 = nn.ReLU(inplace=True)
        # 创建第一个激活层 act1。
        input_channel = output_channel
        # 保存第一个卷积层的输出通道数，作为后续层的输入通道数。
        # building inverted residual blocks 开始构建网络的倒置残差块。
        stages = []
        block = GhostBottleneck
        # 初始化一个列表 stages，用于存储网络的各个阶段。block 变量用于指定倒置残差块的类。
        for cfg in self.cfgs:
            # 遍历网络配置。
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                # 遍历每个配置中的参数，包括扩展比例 k，扩展通道数 exp_size，输出通道数 c，SE 比率 se_ratio 和步长 s。
                output_channel = _make_divisible(c * width, 4)
                # 计算当前 Ghost 瓶颈层的输出通道数。
                hidden_channel = _make_divisible(exp_size * width, 4)
                # 计算当前 Ghost 瓶颈层的隐藏通道数。
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                # 创建一个 Ghost 瓶颈层，并添加到当前阶段的层列表中。
                input_channel = output_channel
                # 更新下一层的输入通道数。
            stages.append(nn.Sequential(*layers))
            # 将当前阶段的所有层打包成一个 Sequential 模块，并添加到 stages 列表中。
        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        self.blocks = nn.Sequential(*stages)
        # 将所有阶段打包成一个 Sequential 模块，形成网络的主要部分。
        # building last several layers 开始构建网络的最后几层。
        output_channel = 1280
        # 设置最后一个卷积层的输出通道数。
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 创建一个全局平均池化层。
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        # 创建最后一个卷积层 conv_head。
        self.act2 = nn.ReLU(inplace=True)
        # 创建第二个激活层 act2。
        self.classifier = nn.Linear(output_channel, num_classes)
        # 创建分类器，一个全连接层，将卷积层的输出映射到类别空间。

    def forward(self, x):
        x = self.conv_stem(x)
        # 将输入 x 通过第一个卷积层。
        x = self.bn1(x)
        # 将结果通过第一个批量归一化层。
        x = self.act1(x)
        # 应用第一个激活函数。
        x = self.blocks(x)
        # 将结果通过所有倒置残差块。
        x = self.global_pool(x)
        # 应用全局平均池化。
        x = self.conv_head(x)
        # 将结果通过最后一个卷积层。
        x = self.act2(x)
        # 应用第二个激活函数。
        x = x.view(x.size(0), -1)
        # 调整张量形状，准备进行全连接层的计算。
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 如果 Dropout 比率大于 0，则应用 Dropout。
        x = self.classifier(x)
        # 将结果通过分类器，得到最终的分类结果。
        return x
        # 返回分类结果。


def ghostnet(**kwargs):
    # 这段代码定义了一个名为 ghostnet 的函数，用于构建 GhostNet 模型。以下是对函数及其参数的详细解释：
    # 定义了一个名为 ghostnet 的函数，它接收任意数量的关键字参数 **kwargs。这些参数将被传递给 GhostNet 类的构造函数。
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    # 定义了一个名为 cfgs 的列表，其中包含了 GhostNet 模型各个阶段的配置。
    # 每个阶段的配置是一个列表，其中每个元素是一个列表，包含以下参数：
    # k：内核大小。
    # t：扩展比例。
    # c：输出通道数。
    # SE：是否使用 Squeeze-and-Excitation (SE) 模块的比率。
    # s：步长。
    # 这些配置定义了 GhostNet 模型的架构。
    return GhostNet(cfgs, **kwargs)
    # 创建并返回一个 GhostNet 类的实例，传入 cfgs 配置和任意数量的关键字参数 **kwargs。


# 整体来看，这段代码演示了如何创建一个 GhostNet 模型，将其设置为评估模式，
# 并使用一个随机初始化的输入张量进行前向传播，最后打印输出张量的大小。这可以用于快速测试模型的行为或进行初步的推理任务。
if __name__=='__main__':
    model = ghostnet()
    # 调用之前定义的 ghostnet 函数来创建一个 GhostNet 模型实例。
    model.eval()
    # 将模型设置为评估模式。这将关闭模型中的 Dropout 和 Batch Normalization 层的训练行为，确保在推理时获得一致的输出。
    print(model)
    # 打印模型的摘要信息，通常包括模型的层数和每层的参数数量。
    input = torch.randn(32, 3, 320, 256)
    # 创建一个随机初始化的输入张量，形状为 (32, 3, 320, 256)。
    # 这里 32 表示批次大小，3 表示颜色通道数，320 和 256 分别表示图像的高度和宽度。
    y = model(input)
    # 将随机初始化的输入张量 input 通过模型 model 进行前向传播，得到输出 y。
    print(y.size())
    # 打印输出张量 y 的大小。这将显示输出张量的形状，
    # 例如 (32, 1000)，其中 1000 通常是模型最后一个全连接层的输出通道数，对应于分类任务中的类别数。
