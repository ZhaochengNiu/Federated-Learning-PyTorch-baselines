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

import torch.optim as toptim
# 导入PyTorch的torch.optim模块，并将其别名设置为toptim。这个模块包含PyTorch提供的所有优化算法。


def sgd(params, optim_args):
    # 定义了一个名为sgd的函数，用于创建随机梯度下降（Stochastic Gradient Descent，SGD）优化器。它接收两个参数：
    # params：模型参数或参数组，这些参数将由优化器进行优化。
    # optim_args：一个包含优化器参数的字典，例如学习率lr。
    return toptim.SGD(params, **optim_args)
    # 使用toptim.SGD创建SGD优化器实例，
    # 并使用解包操作符**将optim_args字典中的关键字参数传递给优化器构造函数。然后返回创建的优化器实例。


def adam(params, optim_args):
    # 定义了一个名为adam的函数，用于创建Adam优化器。它同样接收两个参数：
    # params：模型参数或参数组。
    # optim_args：包含优化器参数的字典。
    return toptim.Adam(params, **optim_args)
    # 使用toptim.Adam创建Adam优化器实例，并使用解包操作符**将optim_args字典中的关键字参数传递给优化器构造函数。
    # 然后返回创建的优化器实例。
