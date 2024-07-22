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

import random
# 导入Python的random模块，这个模块包含用于生成随机数的函数。
import numpy as np
# 导入numpy库，并将其别名设置为np。numpy是一个广泛使用的科学计算库，提供多维数组对象和相应的操作。
import matplotlib.pyplot as plt
# 导入matplotlib.pyplot模块，并将其别名设置为plt。这个模块用于创建图表和绘图。
import torch
# 导入torch库，这是一个开源的机器学习库，用于深度学习。
from torchvision import datasets as tvdatasets, transforms as tvtransforms
# 从torchvision库中导入datasets模块并将其别名设置为tvdatasets，以及导入transforms模块并将其别名设置为tvtransforms。
# 这些模块包含用于加载和处理图像数据集的类和函数。
from torchvision.utils import make_grid
# 从torchvision.utils模块中导入make_grid函数，该函数用于创建一个用于可视化的图像网格。
from sklearn.model_selection import train_test_split
# 从sklearn.model_selection模块中导入train_test_split函数，这个函数用于将数据集分割为训练集和测试集。
import models
# 导入models模块，这可能是一个自定义模块，包含定义的模型架构或模型实例。


# 这个 Subset 类提供了一个灵活的方式来创建原始数据集的子集，可以指定样本索引、增强和归一化变换，并提供了方便的数据集信息摘要。
class Subset(torch.utils.data.Dataset):
    # 这段代码定义了一个名为 Subset 的类，它继承自 torch.utils.data.Dataset，用于创建一个新的数据集，
    # 这个数据集是原始数据集的一个子集，只包含特定的索引 idxs。以下是对类及其方法的逐行解释：
    # 定义了一个名为 Subset 的类，它继承自 PyTorch 的 torch.utils.data.Dataset 类，这是一个用于表示数据集的抽象基类。
    def __init__(self, dataset, idxs, augment=None, normalize=None, name=None):
        # Subset 类的构造函数接收以下参数：
        # dataset：要采样的原始数据集。
        # idxs：要选择的样本索引列表。
        # augment：（可选）应用于数据的增强变换。
        # normalize：（可选）应用于数据的归一化变换。
        # name：（可选）子集数据集的名称。
        self.name = name if name is not None else dataset.name
        # 设置子集数据集的名称，如果提供了 name，则使用它，否则使用原始数据集的名称。
        self.dataset = dataset.dataset if 'dataset' in vars(dataset) else dataset
        # 根据 dataset 是否包含 dataset 属性来确定子集数据集的数据源。
        self.idxs = idxs
        # 保存传入的索引列表。
        self.targets = np.array(dataset.targets)[idxs]
        # 根据索引列表 idxs 获取对应的目标标签。
        self.classes = dataset.classes
        # 保存原始数据集的类别列表。
        if augment is None:
            self.augment = dataset.augment if 'augment' in vars(dataset) else None
        else:
            self.augment = augment
        # 根据是否提供了 augment 参数，确定是否使用原始数据集的增强变换或使用新的增强变换。
        if normalize is None:
            self.normalize = dataset.normalize if 'normalize' in vars(dataset) else None
        else:
            self.normalize = normalize
        # 根据是否提供了 normalize 参数，确定是否使用原始数据集的归一化变换或使用新的归一化变换。

    def __getitem__(self, idx, augmented=True, normalized=True):
        # 定义了 __getitem__ 方法，用于按索引访问子集中的样本。
        # 如果 augmented 为 True，则应用增强变换；如果 normalized 为 True，则应用归一化变换。
        example, target = self.dataset[self.idxs[idx]]
        # 根据索引 idx 和索引列表 self.idxs 从原始数据集中获取样本和目标。
        example = tvtransforms.ToTensor()(example)
        # 将样本转换为 PyTorch 张量。
        if augmented and self.augment is not None:
            example = self.augment(example)
        # 如果需要增强且存在增强变换，则应用它。
        if normalized and self.normalize is not None:
            example = self.normalize(example)
        # 如果需要归一化且存在归一化变换，则应用它。
        return example, target
        # 返回处理后的样本和对应的目标。

    def __len__(self):
        # 定义了 __len__ 方法，返回子集数据集的大小。
        return len(self.targets)
        # 返回目标列表 self.targets 的长度。

    def __str__(self):
        # 定义了 __str__ 方法，返回子集数据集的字符串表示，包含数据集的详细信息。
        dataset_str = f'Name: {self.name}\n'\
                      f'Number of samples: {len(self)}\n'\
                      f'Class distribution: {[(self.targets == c).sum() for c in range(len(self.classes))]}\n'\
                      f'Augmentation: {self.augment}\n'\
                      f'Normalization: {self.normalize}'
        # 格式化字符串，包含数据集名称、样本数量、类别分布、增强变换和归一化变换的信息。
        return dataset_str
        # 返回格式化的字符串。


# 这个 get_mean_std 函数通过遍历整个数据集并计算所有图像的均值和方差，为数据集的归一化提供了所需的统计数据。
# 这对于许多机器学习模型来说是一个重要的预处理步骤，特别是那些对输入数据的分布敏感的模型。
def get_mean_std(dataset, batch_size):
    # 定义了一个名为 get_mean_std 的函数，它接收两个参数：
    # dataset：要处理的数据集。
    # batch_size：在计算均值和标准差时使用的批量大小。
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 创建一个 DataLoader 对象，用于从 dataset 加载数据。
    # 设置 batch_size 参数为指定的批量大小，并设置 shuffle 参数为 False，以确保数据按原始顺序加载。
    total = 0
    mean = 0.
    var = 0.
    # 初始化一些变量，用于存储图像的累积均值、方差和总图像数量。
    for examples, _ in loader:
        # 遍历 DataLoader 中的每个批次。每个批次包含图像 examples 和对应的标签 _（在这里我们不使用标签）。
        # Rearrange batch to be the shape of [B, C, W * H]
        examples = examples.view(examples.size(0), examples.size(1), -1)
        # 将批次中的图像重新排列为 [批量大小，通道数，宽度 * 高度] 的形状。这有助于对每个通道的所有像素值进行操作。
        # Update total number of images
        total += examples.size(0)
        # 更新已处理的图像总数。
        # Compute mean and var here
        mean += examples.mean(2).sum(0)
        # 计算每个通道的均值，并累加到总均值中。mean(2) 计算每个通道的均值，sum(0) 对所有图像的均值求和。
        var += examples.var(2).sum(0)
        # 计算每个通道的方差，并累加到总方差中。var(2) 计算每个通道的方差，sum(0) 对所有图像的方差求和。
    # Final step
    mean /= total
    var /= total
    # 完成所有批次的处理后，将累积均值和方差除以总图像数量，得到最终的均值和标准差。
    return mean.tolist(), torch.sqrt(var).tolist()
    # 将最终的均值和标准差转换为列表，并返回。使用 torch.sqrt 计算方差的平方根，即标准差。


# 这个函数的目的是为模型训练准备数据集，包括加载数据、分割数据集、应用数据增强和归一化等步骤。通过调整 args 参数，可以控制数据集的加载和处理方式。
def get_datasets(name, train_augment, test_augment, args):
    # 定义了一个名为 get_datasets 的函数，它接收以下参数：
    # name：数据集的名称，例如 'CIFAR10' 或 'MNIST'。
    # train_augment：训练数据使用的增强变换。
    # test_augment：测试数据使用的变换（通常是归一化）。
    # args：包含训练参数的命名空间或对象。
    train_tvdataset = getattr(tvdatasets, name)(root='data/'+name, train=True, download=True)
    # 使用 tvdatasets 模块中的相应类创建训练数据集。getattr 函数根据 name 动态获取类，然后使用指定的根目录和训练标志创建实例。
    test_tvdataset = getattr(tvdatasets, name)(root='data/'+name, train=False, download=False)
    # 使用相同的类创建测试数据集，但设置训练标志为 False 并禁止下载（如果数据已存在）。
    # Determine training, validation and test indices
    if args.frac_valid > 0:
        # 如果 args.frac_valid 大于 0，表示需要从训练集中分割出验证集。
        train_idxs, valid_idxs = train_test_split(range(len(train_tvdataset)), test_size=args.frac_valid, stratify=train_tvdataset.targets)
        # 使用 train_test_split 函数从训练集中分割出验证集。根据 args.frac_valid 确定验证集的比例，并确保类别分布一致。
    else:
        train_idxs, valid_idxs = range(len(train_tvdataset)), None
        # 如果没有指定验证集比例，则整个训练集将只用于训练，valid_idxs 设置为 None。
    test_idxs = range(len(test_tvdataset))
    # 创建一个包含测试集所有样本索引的列表。
    # Create training, validation and test datasets
    train_dataset = Subset(dataset=train_tvdataset, idxs=train_idxs, augment=train_augment, name=name)
    # 使用 Subset 类创建训练子集数据集，应用训练增强变换。
    valid_dataset = Subset(dataset=train_tvdataset, idxs=valid_idxs, augment=test_augment, name=name) if valid_idxs is not None else None
    # 如果存在验证集索引，则使用 Subset 类创建验证子集数据集，应用测试变换；否则，将 valid_dataset 设置为 None。
    test_dataset = Subset(dataset=test_tvdataset, idxs=test_idxs, augment=test_augment, name=name)
    # 使用 Subset 类创建测试子集数据集，应用测试变换。
    # Normalization based on pretraining or on previous transforms
    if 'pretrained' in args.model_args and args.model_args['pretrained']:
        # 如果 args.model_args 中包含 'pretrained' 且为 True，则使用预训练模型的均值和标准差。
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # 使用预训练模型的均值和标准差。
    else:
        mean, std = get_mean_std(train_dataset, args.test_bs)
        # 否则，调用 get_mean_std 函数计算训练集的均值和标准差。
    normalize = tvtransforms.Normalize(mean, std)
    # 根据计算得到的均值和标准差创建归一化变换。
    train_dataset.normalize = normalize
    # 将归一化变换应用于训练数据集。
    if valid_dataset is not None: valid_dataset.normalize = normalize
    # 如果存在验证数据集，也将其归一化。
    test_dataset.normalize = normalize
    # 将归一化变换应用于测试数据集。
    return {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}
    # 返回一个字典，包含训练、验证和测试数据集。


# 这个 get_datasets_fig 函数通过可视化的方式展示了不同数据集中原始图像和变换后图像的对比，有助于理解数据增强对图像的影响。
# 通过调整 num_examples 参数，可以控制展示图像的数量。
def get_datasets_fig(datasets, num_examples):
    # 定义了一个名为 get_datasets_fig 的函数，它接收两个参数：
    # datasets：一个字典，包含不同类型（如训练、验证、测试）的数据集。
    # num_examples：每个数据集展示的图像数量。
    types, titles = [], []
    # 初始化两个列表，types 用于存储数据集类型的名称，titles 用于存储数据集类型的标题。
    for type in datasets:
        if datasets[type] is not None:
            types.append(type)
            titles.append(type.capitalize())
    # 遍历 datasets 字典，如果某个类型的数据集不为空，
    # 则将其名称添加到 types 列表，并将其首字母大写的名称添加到 titles 列表。
    fig, ax = plt.subplots(2, len(types))
    # 创建一个图形 fig 和一个包含 2 行 len(types) 列的子图轴对象 ax。
    for i, type in enumerate(types):
        # 遍历 types 列表中的每个数据集类型。
        examples_orig, examples_trans = [], []
        # 初始化两个列表，examples_orig 用于存储原始图像，examples_trans 用于存储变换后的图像。
        for idx in torch.randperm(len(datasets[type]))[:num_examples]:
            # 随机打乱数据集中的索引，并选择前 num_examples 个索引。
            examples_orig.append(datasets[type].__getitem__(idx, augmented=False, normalized=False)[0])
            # 获取原始图像并添加到 examples_orig 列表。
            examples_trans.append(datasets[type].__getitem__(idx, augmented=True, normalized=False)[0])
            # 获取变换后的图像并添加到 examples_trans 列表。
        examples_orig = torch.stack(examples_orig)
        # 将原始图像列表堆叠成一个张量。
        examples_trans = torch.stack(examples_trans)
        # 将变换后的图像列表堆叠成一个张量。
        grid_orig = np.transpose(make_grid(examples_orig, nrow=int(num_examples**0.5)).numpy(), (1,2,0))
        # 使用 make_grid 函数创建原始图像的网格，并使用 np.transpose 调整维度顺序。
        grid_trans = np.transpose(make_grid(examples_trans, nrow=int(num_examples**0.5)).numpy(), (1,2,0))
        # 使用 make_grid 函数创建变换后图像的网格，并使用 np.transpose 调整维度顺序。
        ax[0, i].imshow(grid_orig)
        # 在第一个子图中展示原始图像网格。
        ax[0, i].set_title(titles[i] + ' original')
        # 设置第一个子图的标题。
        ax[1, i].imshow(grid_trans)
        # 在第二个子图中展示变换后图像网格。
        ax[1, i].set_title(titles[i] + ' transformed')
        # 设置第二个子图的标题。
    fig.tight_layout()
    # 自动调整子图参数，使之填充整个图形区域。
    fig.set_size_inches(4*len(types), 8)
    # 设置图形的大小。
    return fig
    # 返回创建的图形对象。

