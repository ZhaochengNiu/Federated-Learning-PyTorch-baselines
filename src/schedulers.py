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

from torch.optim import lr_scheduler
# 从 PyTorch 的 optim 模块导入 lr_scheduler，这个模块包含 PyTorch 内置的学习率调度器。
from utils import Scheduler
# 从 utils 模块导入 Scheduler 类，这可能是一个自定义的基类，用于学习率调度器的实现。


class fixed(Scheduler):
    # 定义了一个名为 fixed 的类，继承自 Scheduler 类。这个类可能表示一个固定学习率的调度器。
    def __init__(self, optimizer, sched_args):
        # fixed 类的构造函数接收两个参数：
        # optimizer：PyTorch 优化器实例。
        # sched_args：调度器参数的字典。
        self.name = 'FixedLR'
        # 设置调度器的名称为 'FixedLR'。
        self.optimizer = optimizer
        # 将传入的优化器实例保存在 self.optimizer 中。

    def get_lr(self):
        # 定义了一个 get_lr 方法，用于获取当前学习率。
        return [group['lr'] for group in self.optimizer.param_groups]
        # 返回优化器参数组中的学习率列表。

    def step(self):
        pass

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass


class step(lr_scheduler.StepLR, Scheduler):
    # 定义了一个名为 step 的类，继承自 PyTorch 的 lr_scheduler.StepLR 和 Scheduler 类。
    # 这个类实现了步进学习率调度策略。
    def __init__(self, optimizer, sched_args):
        # step 类的构造函数接收两个参数：
        # optimizer：PyTorch 优化器实例。
        # sched_args：调度器参数的字典。
        super(step, self).__init__(optimizer, **sched_args)
        # 调用父类 lr_scheduler.StepLR 的构造函数，初始化步进学习率调度器。
        self.name = 'StepLR'
        # 设置调度器的名称为 'StepLR'。


class const(lr_scheduler.ConstantLR, Scheduler):
    # 定义了一个名为 const 的类，继承自 PyTorch 的 lr_scheduler.ConstantLR 和 Scheduler 类。
    # 这个类实现了恒定学习率调度策略。
    def __init__(self, optimizer, sched_args):
        # const 类的构造函数接收两个参数：
        # optimizer：PyTorch 优化器实例。
        # sched_args：调度器参数的字典。
        super(const, self).__init__(optimizer, **sched_args)
        # 调用父类 lr_scheduler.ConstantLR 的构造函数，初始化恒定学习率调度器。
        self.name = 'ConstantLR'
        # 设置调度器的名称为 'ConstantLR'。


class plateau_loss(lr_scheduler.ReduceLROnPlateau, Scheduler):
    # 定义了一个名为 plateau_loss 的类，继承自 PyTorch 的 lr_scheduler.ReduceLROnPlateau 和 Scheduler 类。
    # 这个类实现了基于性能高原（plateau）的学习率衰减策略。
    def __init__(self, optimizer, sched_args):
        # plateau_loss 类的构造函数接收两个参数：
        # optimizer：PyTorch 优化器实例。
        # sched_args：调度器参数的字典。
        super(plateau_loss, self).__init__(optimizer, **sched_args)
        # 调用父类 lr_scheduler.ReduceLROnPlateau 的构造函数，初始化基于性能高原的学习率衰减调度器。
        self.name = 'ReduceLROnPlateauLoss'
        # 设置调度器的名称为 'ReduceLROnPlateauLoss'。
