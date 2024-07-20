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

from copy import deepcopy
# 从 Python 标准库的 copy 模块导入 deepcopy 函数，该函数用于创建对象的深拷贝。
import matplotlib.pyplot as plt
# 导入 matplotlib.pyplot，用于绘图。
import numpy as np
# 导入 numpy，用于数值计算。
import torch
# 导入 torch，用于深度学习相关操作。
from torch.nn import CrossEntropyLoss
# 从 torch.nn 模块导入 CrossEntropyLoss，用于计算交叉熵损失，常用于多分类问题。
from torch.utils.data import DataLoader, Dataset, Subset
# 从 torch.utils.data 模块导入 DataLoader、Dataset 和 Subset，这些用于数据加载和处理。
from utils import inference
# 从 utils 模块导入 inference，这个函数可能用于模型推理。


class Client(object):
    def __init__(self, args, datasets, idxs):
        # Client 类的构造函数接收三个参数：
        # args：包含客户端参数的命名空间或对象。
        # datasets：包含训练、验证和测试数据集的字典。
        # idxs：包含训练、验证和测试数据索引的字典。
        self.args = args
        # 将传入的 args 赋值给实例变量 self.args。
        # Create dataloaders
        self.train_bs = self.args.train_bs if self.args.train_bs > 0 else len(idxs['train'])
        # 根据 args.train_bs 的值设置训练时的批量大小，如果 args.train_bs 不大于0，则使用训练数据索引的长度作为批量大小。
        self.loaders = {}
        # 初始化一个空字典 self.loaders，用于存储数据加载器。
        self.loaders['train'] = DataLoader(Subset(datasets['train'], idxs['train']), batch_size=self.train_bs, shuffle=True) if len(idxs['train']) > 0 else None
        # 创建训练数据加载器。如果 idxs['train'] 不为空，则使用 DataLoader 和 Subset 来加载训练数据。
        self.loaders['valid'] = DataLoader(Subset(datasets['valid'], idxs['valid']), batch_size=args.test_bs, shuffle=False) if idxs['valid'] is not None and len(idxs['valid']) > 0 else None
        # 创建验证数据加载器。如果 idxs['valid'] 不为空，则使用 DataLoader 和 Subset 来加载验证数据。
        self.loaders['test'] = DataLoader(Subset(datasets['test'], idxs['test']), batch_size=args.test_bs, shuffle=False) if len(idxs['test']) > 0 else None
        # 创建测试数据加载器。如果 idxs['test'] 不为空，则使用 DataLoader 和 Subset 来加载测试数据。
        # Set criterion
        if args.fedir:
            # 检查是否使用联邦重要性重加权（FedIR）。
            # Importance Reweighting (FedIR)
            labels = set(datasets['train'].targets)
            p = torch.tensor([(torch.tensor(datasets['train'].targets) == label).sum() for label in labels]) / len(datasets['train'].targets)
            q = torch.tensor([(torch.tensor(datasets['train'].targets)[idxs['train']] == label).sum() for label in labels]) / len(torch.tensor(datasets['train'].targets)[idxs['train']])
            weight = p/q
            # 如果使用 FedIR，则计算每个类别的权重。p 是全局分布，q 是本地分布，权重计算为 p/q。
        else:
            # No Importance Reweighting
            weight = None
            # 如果不使用 FedIR，则将权重设置为 None。
        self.criterion = CrossEntropyLoss(weight=weight)
        # 创建一个 CrossEntropyLoss 损失函数实例，传入计算得到的权重或 None。

    def train(self, model, optim, device):
        # Drop client if train set is empty
        # train 方法接收三个参数：
        # model：要训练的模型。
        # optim：用于优化模型的优化器。
        # device：模型和数据所在的设备（CPU或GPU）。
        if self.loaders['train'] is None:
            if not self.args.quiet: print(f'            No data!')
            return None, 0, 0, None
        # 如果训练数据加载器为空（即没有训练数据），则打印消息并返回 None。
        # Determine if client is a straggler and drop it if required
        straggler = np.random.binomial(1, self.args.hetero)
        # 根据 self.args.hetero 决定客户端是否是落后者（straggler）。这是通过从伯努利分布中随机抽取来模拟的。
        if straggler and self.args.drop_stragglers:
            if not self.args.quiet: print(f'            Dropped straggler!')
            return None, 0, 0, None
        # 如果客户端是落后者且参数 self.args.drop_stragglers 为 True，则打印消息并返回 None。
        epochs = np.random.randint(1, self.args.epochs) if straggler else self.args.epochs
        # 确定训练轮数。如果是落后者，则从 1 到 self.args.epochs 随机选择；否则使用全部轮数。
        # Create training loader
        if self.args.vc_size is not None:
            # 检查是否使用虚拟客户端（FedVC）。
            # Virtual Client (FedVC)
            if len(self.loaders['train'].dataset) >= self.args.vc_size:
                train_idxs_vc = torch.randperm(len(self.loaders['train'].dataset))[:self.args.vc_size]
            else:
                train_idxs_vc = torch.randint(len(self.loaders['train'].dataset), (self.args.vc_size,))
            train_loader = DataLoader(Subset(self.loaders['train'].dataset, train_idxs_vc), batch_size=self.train_bs, shuffle=True)
        else:
            # No Virtual Client
            train_loader = self.loaders['train']
            # 如果不使用 FedVC，则使用原始的训练数据加载器。
        client_stats_every = self.args.client_stats_every if self.args.client_stats_every > 0 and self.args.client_stats_every < len(train_loader) else len(train_loader)
        # 确定客户端统计信息打印的频率。
        # Train new model
        model.to(device)
        self.criterion.to(device)
        # 将模型和损失函数的参数移动到指定的设备。
        model.train()
        # 将模型设置为训练模式。
        model_server = deepcopy(model)
        # 创建模型的深拷贝，用于计算模型更新。
        iter = 0
        for epoch in range(epochs):
            # 开始训练循环。
            loss_sum, loss_num_images, num_images = 0., 0, 0
            # 初始化损失累计和统计变量。
            for batch, (examples, labels) in enumerate(train_loader):
                # 遍历训练数据加载器中的批次。
                examples, labels = examples.to(device), labels.to(device)
                # 将数据和标签移动到指定的设备。
                model.zero_grad()
                log_probs = model(examples)
                loss = self.criterion(log_probs, labels)
                # 执行前向传播，计算损失。
                if self.args.mu > 0 and epoch > 0:
                    # 如果使用 FedProx（参数 self.args.mu 大于 0），则在损失中添加近端项。
                    # Add proximal term to loss (FedProx)
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(model.parameters(), model_server.parameters()):
                        w_diff += torch.pow(torch.norm(w.data - w_t.data), 2)
                        #w.grad.data += self.args.mu * (w.data - w_t.data)
                        w.grad.data += self.args.mu * (w_t.data - w.data)
                    # 计算模型参数与服务器模型参数的差异，并更新梯度。
                    loss += self.args.mu / 2. * w_diff
                loss_sum += loss.item() * len(labels)
                loss_num_images += len(labels)
                num_images += len(labels)
                # 更新损失累计和统计变量。
                loss.backward()
                optim.step()
                # 执行反向传播和优化器步骤。
                # After client_stats_every batches...
                if (batch + 1) % client_stats_every == 0:
                    # 检查是否达到打印统计信息的频率。
                    # ...Compute average loss
                    loss_running = loss_sum / loss_num_images
                    # 计算运行平均损失。
                    # ...Print stats
                    if not self.args.quiet:
                        print('            ' + f'Epoch: {epoch+1}/{epochs}, '\
                                               f'Batch: {batch+1}/{len(train_loader)} (Image: {num_images}/{len(train_loader.dataset)}), '\
                                               f'Loss: {loss.item():.6f}, ' \
                                               f'Running loss: {loss_running:.6f}')
                    # 打印训练进度。
                    loss_sum, loss_num_images = 0., 0
                    # 重置损失累计和统计变量。
                iter += 1

        # Compute model update
        model_update = {}
        for key in model.state_dict():
            model_update[key] = torch.sub(model_server.state_dict()[key], model.state_dict()[key])
        # 计算模型更新。
        return model_update, len(train_loader.dataset), iter, loss_running
        # 返回模型更新、数据集大小、迭代次数和运行平均损失。
    def inference(self, model, type, device):
        # inference 方法接收三个参数：
        # model：要进行推理的模型。
        # type：数据集类型（'train'、'valid' 或 'test'）。
        # device：模型和数据所在的设备（CPU或GPU）。
        return inference(model, self.loaders[type], device)
        # 调用 inference 函数进行模型推理，返回推理结果。

