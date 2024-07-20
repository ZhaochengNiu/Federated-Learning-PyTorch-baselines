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

import io, re
# 导入Python标准库中的io模块，提供输入输出功能，以及re模块，提供正则表达式的功能。
from copy import deepcopy
# 从copy模块导入deepcopy函数，用于创建对象的深拷贝。
from contextlib import redirect_stdout
# 从 contextlib 模块导入 redirect_stdout 函数，该函数可以重定向标准输出到另一个文件对象。
import torch
# 导入 torch 库，这是一个广泛使用的开源机器学习库，基于 Torch。
from torch.nn import CrossEntropyLoss
# 从 torch.nn 模块导入 CrossEntropyLoss，用于计算交叉熵损失，常用于多分类问题。
from torchinfo import summary
# 导入 torchinfo 模块的 summary 函数，该函数可以打印出模型的摘要信息，包括层的输出尺寸、参数数量等。
import optimizers, schedulers
# 导入自定义模块 optimizers 和 schedulers，这些模块可能包含优化器和学习率调度器的实现。

types_pretty = {'train': 'training', 'valid': 'validation', 'test': 'test'}
# 定义一个字典 types_pretty，将数据集类型（如'train'、'valid'、'test'）映射为更易读的字符串（如'training'、'validation'、'test'）。
# 这可以用于打印或显示更友好的数据集描述。


class Scheduler():
    def __str__(self):
        # 定义了 __str__ 方法，这是 Python 中的特殊方法，用于定义对象的字符串表示。
        # 当打印 Scheduler 类的实例或使用 str() 函数时，会调用这个方法。
        sched_str = '%s (\n' % self.name
        # 初始化一个字符串 sched_str，它将包含调度器的字符串表示。使用 self.name 属性作为开头，这可能是调度器的名称。
        for key in vars(self).keys():
            # 遍历 self（即 Scheduler 实例）的所有属性的键。
            if key != 'name':
                # 跳过 name 属性，因为它已经在开头处理过了。
                value = vars(self)[key]
                # 获取当前属性键 key 对应的值。
                if key == 'optimizer': value = str(value).replace('\n', '\n        ').replace('    )', ')')
                # 如果当前属性是 optimizer，对它的字符串表示进行格式化，使其在输出中有更好的缩进和格式。
                sched_str +=  '    %s: %s\n' % (key, value)
                # 将当前属性的键和值格式化为字符串，并添加到 sched_str。
        sched_str += ')'
        # 在字符串表示的末尾添加一个闭合括号。
        return sched_str
        # 返回完整的调度器字符串表示。


def average_updates(w, n_k):
    # 定义了一个名为 average_updates 的函数，它接收两个参数：
    # w：一个字典列表，其中每个字典包含一个客户端模型的权重。
    # n_k：一个权重列表，表示每个客户端的权重或贡献度。
    w_avg = deepcopy(w[0])
    # 创建一个新的字典 w_avg，它是列表 w 中第一个字典的深拷贝。这将作为聚合权重的基础。
    for key in w_avg.keys():
        # 遍历 w_avg 字典中的所有键（即模型参数的名称）。
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        # 对于每个参数 key，将 w_avg[key] 与第一个客户端的权重 n_k[0] 相乘。这为后续的加权和计算做准备。
        for i in range(1, len(w)):
            # 从第二个客户端开始，遍历所有客户端。
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
            # 将每个客户端的参数 w[i][key] 乘以其权重 n_k[i]，然后加到 w_avg[key] 上。alpha 参数用于指定缩放因子。
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
        # 在累加完所有客户端的加权参数后，将 w_avg[key] 除以所有权重的总和 sum(n_k)，得到平均权重。
    return w_avg
    # 返回计算得到的平均权重字典 w_avg。


def inference(model, loader, device):
    # 定义了一个名为 inference 的函数，它接收三个参数：
    # model：要评估的模型。
    # loader：包含数据的 DataLoader 对象。
    # device：模型和数据所在的设备（CPU或GPU）。
    if loader is None:
        return None, None
    # 如果传入的 loader 为 None，则直接返回 None, None，表示没有数据进行评估。
    criterion = CrossEntropyLoss().to(device)
    # 创建一个 CrossEntropyLoss 损失函数，并将其移动到指定的设备。
    loss, total, correct = 0., 0, 0
    # 初始化损失 loss、总样本数 total 和正确预测数 correct。
    model.eval()
    # 将模型设置为评估模式，这会关闭 Dropout 和 BatchNorm 层的训练行为。
    with torch.no_grad():
        # 进入 torch.no_grad() 上下文管理器，这将关闭梯度计算，从而减少内存消耗并加速推理。
        for batch, (examples, labels) in enumerate(loader):
            # 遍历 loader 中的每个批次。
            examples, labels = examples.to(device), labels.to(device)
            # 将数据和标签移动到指定的设备。
            log_probs = model(examples)
            # 执行模型的前向传播，计算输入数据 examples 的对数概率。
            loss += criterion(log_probs, labels).item() * len(labels)
            # 计算当前批次的损失，并将损失乘以批次大小后累加到总损失。
            _, pred_labels = torch.max(log_probs, 1)
            # 从模型输出的对数概率中找到最大概率对应的预测标签。
            pred_labels = pred_labels.view(-1)
            # 将预测标签展平为一维。
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            # 计算预测标签与真实标签相同的数量，累加到 correct。
            total += len(labels)
            # 累加当前批次的样本总数到 total。
    accuracy = correct/total
    # 计算总准确率，即正确预测数除以总样本数。
    loss /= total
    # 计算平均损失，即总损失除以总样本数。
    return accuracy, loss
    # 返回评估得到的准确率和平均损失。


def get_acc_avg(acc_types, clients, model, device):
    # 定义了一个名为 get_acc_avg 的函数，它接收四个参数：
    # acc_types：一个字符串列表，指定要计算准确率的数据集类型（例如 ['train', 'test']）。
    # clients：一个客户端列表，每个客户端包含其数据集和模型。
    # model：要评估的模型。
    # device：模型和数据所在的设备（CPU或GPU）。
    acc_avg = {}
    # 初始化一个字典 acc_avg，用于存储每个数据集类型的平均准确率。
    for type in acc_types:
        # 遍历 acc_types 列表中的每个数据集类型。
        acc_avg[type] = 0.
        num_examples = 0
        # 对于每个数据集类型，初始化该类型的平均准确率 acc_avg[type] 和样本数量 num_examples。
        for client_id in range(len(clients)):
            # 遍历所有客户端。
            acc_client, _ = clients[client_id].inference(model, type=type, device=device)
            # 对于每个客户端，调用其 inference 方法进行模型推理，并获取准确率 acc_client。这里传入模型、数据集类型和设备作为参数。
            if acc_client is not None:
                # 检查获取的准确率是否不为 None。
                acc_avg[type] += acc_client * len(clients[client_id].loaders[type].dataset)
                # 如果准确率不为 None，则将其乘以该客户端的数据集大小，并累加到该类型的平均准确率中。
                num_examples += len(clients[client_id].loaders[type].dataset)
                # 累加该客户端的数据集大小到 num_examples。
        acc_avg[type] = acc_avg[type] / num_examples if num_examples != 0 else None
        # 计算最终的平均准确率，即累加的准确率除以样本总数。如果 num_examples 为0（即没有样本），则将平均准确率设置为 None。
    return acc_avg
    # 返回包含每个数据集类型平均准确率的字典。


def printlog_stats(quiet, logger, loss_avg, acc_avg, acc_types, lr, round, iter, iters):
    # 定义了一个名为 printlog_stats 的函数，它接收以下参数：
    # quiet：一个布尔值，指示是否在控制台打印详细信息。
    # logger：一个日志记录器对象，用于记录训练统计信息。
    # loss_avg：平均损失。
    # acc_avg：一个字典，包含不同数据集类型的平均准确率。
    # acc_types：一个字符串列表，指定要记录准确率的数据集类型（例如 ['train', 'test']）。
    # lr：当前的学习率。
    # round：当前的训练轮次。
    # iter：当前的迭代次数。
    # iters：总的迭代次数。
    if not quiet:
        # 如果 quiet 为 False，则在控制台打印详细信息。
        print(f'        Iteration: {iter}', end='')
        # 打印当前的迭代次数。
        if iters is not None: print(f'/{iters}', end='')
        # 如果总迭代次数 iters 不为 None，则打印总迭代次数。
        print()
        print(f'        Learning rate: {lr}')
        # 打印当前的学习率。
        print(f'        Average running loss: {loss_avg:.6f}')
        # 打印平均运行损失，保留6位小数。
        for type in acc_types:
            # 遍历 acc_types 列表中的每个数据集类型。
            print(f'        Average {types_pretty[type]} accuracy: {acc_avg[type]:.3%}')
            # 打印每个数据集类型的平均准确率，保留3位小数，并使用 types_pretty 字典将数据集类型转换为更易读的字符串。
    if logger is not None:
        # 如果提供了日志记录器 logger，则执行以下记录操作。
        logger.add_scalar('Learning rate (Round)', lr, round)
        # 记录当前轮次的学习率。
        logger.add_scalar('Learning rate (Iteration)', lr, iter)
        # 记录当前迭代次数的学习率。
        logger.add_scalar('Average running loss (Round)', loss_avg, round)
        # 记录当前轮次的平均运行损失。
        logger.add_scalar('Average running loss (Iteration)', loss_avg, iter)
        # 记录当前迭代次数的平均运行损失。
        for type in acc_types:
            # 遍历 acc_types 列表中的每个数据集类型。
            logger.add_scalars('Average accuracy (Round)', {types_pretty[type].capitalize(): acc_avg[type]}, round)
            # 记录当前轮次的平均准确率。
            logger.add_scalars('Average accuracy (Iteration)', {types_pretty[type].capitalize(): acc_avg[type]}, iter)
            # 记录当前迭代次数的平均准确率。
        logger.flush()
        # 刷新日志记录器，确保所有记录的信息被写入。


def exp_details(args, model, datasets, splits):
    # 定义了一个名为 exp_details 的函数，它接收四个参数：
    # args：包含实验配置的参数。
    # model：要训练和评估的模型。
    # datasets：包含训练、验证和测试数据集的字典。
    # splits：包含数据集拆分的字典。
    if args.device == 'cpu':
        device = 'CPU'
    else:
        device = str(torch.cuda.get_device_properties(args.device))
        device = (', ' + re.sub('_CudaDeviceProperties\(|\)', '', device)).replace(', ', '\n            ')
    # 根据 args.device 的值，确定设备类型并格式化设备信息字符串。
    input_size = (args.train_bs,) + tuple(datasets['train'][0][0].shape)
    # 获取训练数据集中第一个样本的形状，并将其与训练批量大小结合，用于模型摘要的输入尺寸。
    summ = str(summary(model, input_size, depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device=args.device))
    # 使用 summary 函数生成模型摘要，显示模型的输出尺寸、卷积核尺寸、参数数量和乘法累加次数。
    summ = '        ' + summ.replace('\n', '\n        ')

    optimizer = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    # 根据 args.optim 参数获取优化器类，并创建优化器实例。
    scheduler = getattr(schedulers, args.sched)(optimizer, args.sched_args)
    # 根据 args.sched 参数获取调度器类，并创建调度器实例。
    if args.centralized:
        algo = 'Centralized'
    else:
        if args.fedsgd:
            algo = 'FedSGD'
        else:
            algo = 'FedAvg'
        if args.server_momentum:
            algo += 'M'
        if args.fedir:
            algo += ' + FedIR'
        if args.vc_size is not None:
            algo += ' + FedVC'
        if args.mu:
            algo += ' + FedProx'
        if args.drop_stragglers:
            algo += ' (Drop Stragglers)'
        # 根据 args 参数构建算法名称字符串，包括联邦学习算法和附加的技术。
    f = io.StringIO()
    with redirect_stdout(f):
        print('Experiment summary:')
        print(f'    Algorithm:')
        print(f'        Algorithm: {algo}')
        print(f'        ' + (f'Rounds: {args.rounds}' if args.iters is None else f'Iterations: {args.iters}'))
        print(f'        Clients: {args.num_clients}')
        print(f'        Fraction of clients: {args.frac_clients}')
        print(f'        Client epochs: {args.epochs}')
        print(f'        Training batch size: {args.train_bs}')
        print(f'        System heterogeneity: {args.hetero}')
        print(f'        Server learning rate: {args.server_lr}')
        print(f'        Server momentum (FedAvgM): {args.server_momentum}')
        print(f'        Virtual client size (FedVC): {args.vc_size}')
        print(f'        Mu (FedProx): {args.mu}')
        print()

        print('    Dataset and split:')
        print('        Training set:')
        print('            ' + str(datasets['train']).replace('\n','\n            '))
        if datasets['valid'] is not None:
            print('        Validation set:')
            print('            ' + str(datasets['valid']).replace('\n','\n            '))
        print('        Test set:')
        print('            ' + str(datasets['test']).replace('\n','\n            '))
        print(f'        Identicalness: {args.iid} (EMD = {splits["train"].emd["class"]})')
        print(f'        Balance: {args.balance} (EMD = {splits["train"].emd["client"]})')
        print()

        print('    Scheduler: %s' % (str(scheduler).replace('\n', '\n    ')))
        print()

        print('    Model:')
        print(summ)
        print()

        print('    Other:')
        print(f'        Test batch size: {args.test_bs}')
        print(f'        Random seed: {args.seed}')
        print(f'        Device: {device}')

    return f.getvalue()
