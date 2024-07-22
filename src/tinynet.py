# 2021.01.09-Changed for main script for testing TinyNet on ImageNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
An implementation of TinyNet

Requirements: timm==0.1.20
"""
from timm.models.efficientnet_builder import *
# 这行代码从timm.models.efficientnet_builder模块导入所有公开的（不以下划线开头的）函数、类和变量。
# efficientnet_builder模块可能包含用于构建或自定义EfficientNet架构的工具和方法。
from timm.models.efficientnet import EfficientNet, EfficientNetFeatures, _cfg
# 这行代码从timm.models.efficientnet模块中导入了三个具体的项：
# EfficientNet：这是一个类，代表EfficientNet模型的主要架构。
# EfficientNetFeatures：这可能是一个类，用于从EfficientNet模型中提取特征，用于特征提取或中间层的访问。
# _cfg：这是一个以单下划线开头的函数或变量，表明它可能是一个内部使用的配置或辅助工具，用于配置EfficientNet模型的参数。
from timm.models.registry import register_model
# 这行代码从timm.models.registry模块中导入了register_model函数。
# register_model函数用于向模型注册表中添加新的模型类，这样它们就可以通过名称被检索和实例化。
from timm.models.layers import Swish
# 这行代码从timm.models.layers模块中导入了Swish类。
# Swish是一个激活函数类，代表Swish激活函数，这是一种在深度学习中使用的非单调激活函数，特别是在EfficientNet架构中。


def _gen_tinynet(variant_cfg, channel_multiplier=1.0, depth_multiplier=1.0, depth_trunc='round', **kwargs):
    """Creates a TinyNet model.
    """
    # 定义了一个名为 _gen_tinynet 的函数，它接收以下参数：
    # variant_cfg：变体配置，通常包含模型的默认配置信息。
    # channel_multiplier：通道乘数，默认为 1.0，用于调整模型的通道数。
    # depth_multiplier：深度乘数，默认为 1.0，用于调整模型的深度。
    # depth_trunc：深度截断策略，默认为 'round'，用于确定如何根据深度乘数调整层的深度。
    # **kwargs：额外的关键字参数，用于传递其他配置项。
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'], ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'], ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'], ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    # 定义了一个名为 arch_def 的列表，其中包含模型架构的字符串描述。
    # 每个字符串描述了一个卷积块的配置，包括卷积类型、内核大小、步长、扩展比、通道数和是否使用SE（Squeeze-and-Excitation）模块。
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc=depth_trunc),
        num_features=max(1280, round_channels(1280, channel_multiplier, 8, None)),
        stem_size=32,
        fix_stem=True,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    # 创建一个名为 model_kwargs 的字典，包含用于初始化 EfficientNet 模型的参数：
    # block_args：通过 decode_arch_def 函数解析 arch_def 列表，根据深度乘数和深度截断策略生成每个块的参数。
    # num_features：模型中特征图的最大通道数，通过 round_channels 函数计算。
    # stem_size：Stem 层的输出通道数。
    # fix_stem：是否固定 Stem 层的输出通道数。
    # channel_multiplier：通道乘数。
    # act_layer：激活层，这里使用 Swish 激活函数。
    # norm_kwargs：批量归一化（BatchNorm）的参数，通过 resolve_bn_args 函数解析 kwargs 生成。
    # **kwargs：其他额外的关键字参数。
    model = EfficientNet(**model_kwargs)
    # 使用 model_kwargs 字典中的参数初始化一个 EfficientNet 模型实例。
    model.default_cfg = variant_cfg
    # 为模型实例设置默认配置，这通常包含模型的输入尺寸、输出尺寸和其他一些预设的配置。
    return model
    # 返回创建的 EfficientNet 模型实例。


@register_model
def tinynet(r=1.0, w=1.0, d=1.0, **kwargs):
    # 这段代码演示了如何在 Python 中使用装饰器 @register_model 来注册一个新的模型函数，
    # 然后定义了这个函数 tinynet 来创建并返回一个 TinyNet 模型。以下是对代码的逐行解释：
    # 定义了一个名为 tinynet 的函数，它接收以下参数：
    # r：分辨率乘数，默认为 1.0，用于调整模型输入图像的分辨率。
    # w：宽度乘数，默认为 1.0，用于调整模型的通道数。
    # d：深度乘数，默认为 1.0，用于调整模型的深度。
    # **kwargs：额外的关键字参数，用于传递其他配置项。
    """ TinyNet """
    hw = int(224 * r)
    # 计算调整后的模型输入高度和宽度。原始尺寸为 224x224 像素，根据分辨率乘数 r 进行缩放。
    model = _gen_tinynet(_cfg(input_size=(3, hw, hw)), channel_multiplier=w, depth_multiplier=d, **kwargs)
    # 调用之前定义的 _gen_tinynet 函数来创建 TinyNet 模型。传递以下参数：
    # _cfg(input_size=(3, hw, hw))：配置对象，指定了模型的输入尺寸，这里为三通道的 hw x hw 像素图像。
    # channel_multiplier=w：通道乘数，用于调整模型的通道数。
    # depth_multiplier=d：深度乘数，用于调整模型的深度。
    # **kwargs：其他额外的关键字参数。
    return model
    # 返回创建的 TinyNet 模型实例。
