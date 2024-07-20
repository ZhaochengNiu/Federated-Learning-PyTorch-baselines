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

import torchvision.transforms as tvtransforms
# 导入 torchvision.transforms 模块，这是一个常用于图像处理的模块，提供了一系列图像变换操作。
from datasets_utils import get_datasets
# 从 datasets_utils 模块导入 get_datasets 函数，这个函数可能用于获取和处理数据集。


def cifar10(args, dataset_args):
    # 定义了一个名为 cifar10 的函数，它接收两个参数：
    # args：包含模型训练参数的参数对象。
    # dataset_args：包含数据集特定参数的字典。
    if 'augment' in dataset_args and not dataset_args['augment']:
        # 检查 dataset_args 字典中是否存在键 'augment' 并且其值为 False。这通常用于控制是否应用数据增强。
        train_augment, test_augment = None, None
        # 如果不进行数据增强，则将训练和测试的变换设置为 None。
    else:
        # 如果进行数据增强：
        train_augment = tvtransforms.Compose([
            tvtransforms.RandomCrop(24),
            tvtransforms.RandomHorizontalFlip(),
            tvtransforms.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        # 定义训练数据的增强变换，包括：
        # 随机裁剪图片到 24x24 像素。
        # 随机水平翻转图片。
        # 随机调整图片的亮度和对比度。
        test_augment = tvtransforms.CenterCrop(24)
        # 定义测试数据的变换，使用中心裁剪到 24x24 像素。
    return get_datasets(name='CIFAR10', train_augment=train_augment, test_augment=test_augment, args=args)
    # 调用 get_datasets 函数获取 CIFAR-10 数据集，传入参数：
    # name：数据集的名称，这里是 'CIFAR10'。
    # train_augment：训练数据的变换。
    # test_augment：测试数据的变换。
    # args：模型训练参数。


def mnist(args):
    train_augment, test_augment = None, None
    # 在 MNIST 数据集的处理中，不使用任何数据增强。因此，训练和测试的变换（augmentation）都被设置为 None。
    return get_datasets(name='MNIST', train_augment=train_augment, test_augment=test_augment, args=args)
    # 调用 get_datasets 函数来获取 MNIST 数据集。传入的参数包括：
    # name：数据集的名称，这里指定为 'MNIST'。
    # train_augment：训练数据的变换，这里为 None，表示不进行数据增强。
    # test_augment：测试数据的变换，这里为 None，表示不进行数据增强。
    # args：模型训练参数。


def fmnist(args):
    train_augment, test_augment = None, None
    # 在 FashionMNIST 数据集的处理中，同样不使用任何数据增强。因此，训练和测试的变换都被设置为 None。
    return get_datasets(name='FashionMNIST', train_augment=train_augment, test_augment=test_augment, args=args)
    # 调用 get_datasets 函数来获取 FashionMNIST 数据集。传入的参数包括：
    # name：数据集的名称，这里指定为 'FashionMNIST'。
    # train_augment：训练数据的变换，这里为 None，表示不进行数据增强。
    # test_augment：测试数据的变换，这里为 None，表示不进行数据增强。
    # args：模型训练参数。

