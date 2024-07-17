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

import random, re
# 导入Python标准库中的random模块，提供生成随机数的函数；re模块，提供正则表达式的功能。
from copy import deepcopy
# 从copy模块导入deepcopy函数，用于深度复制对象。
from os import environ
# 从os模块导入environ属性，提供访问环境变量的功能。
from time import time
# 从time模块导入time函数，用于获取当前时间的时间戳。
from datetime import timedelta
# 从datetime模块导入timedelta类，用于表示两个日期或时间之间的差异。
from collections import defaultdict
# 从collections模块导入defaultdict类，提供默认值的字典。
import numpy as np
# 导入numpy库，并简称为np，这是一个广泛使用的科学计算库，提供多维数组对象和相应的操作。
import torch
# 导入torch库，PyTorch是一个开源的机器学习库，基于Torch。
from torch.utils.data import DataLoader
# 从torch.utils.data模块导入DataLoader类，用于加载数据集。
from torch.utils.tensorboard import SummaryWriter
# 从torch.utils.tensorboard模块导入SummaryWriter类，用于将训练过程中的信息写入TensorBoard。

import datasets, models, optimizers, schedulers
# 导入自定义模块datasets、models、optimizers和schedulers，这些模块可能包含数据集加载、模型定义、优化器和学习率调度器的实现。
from options import args_parser
# 从options模块导入args_parser函数，这个函数可能用于解析命令行参数或配置文件。
from utils import average_updates, exp_details, get_acc_avg, printlog_stats
# 从utils模块导入一些实用函数，可能包括平均更新、实验细节、准确率平均值获取和打印日志统计信息。
from datasets_utils import Subset, get_datasets_fig
# 从datasets_utils模块导入Subset类和get_datasets_fig函数，可能用于数据集的子集操作和数据集相关图表的获取。
from sampling import get_splits, get_splits_fig
# 从sampling模块导入get_splits函数和get_splits_fig函数，可能用于数据拆分和拆分相关图表的获取。
from client import Client
# 从client模块导入Client类，这个类可能定义了客户端的行为，用于模拟联邦学习中的客户端。

if __name__ == '__main__':
    # Start timer
    # 开始计时，记录脚本开始执行的时间。
    start_time = time()

    # Parse arguments and create/load checkpoint
    args = args_parser()
    # 解析命令行参数，并根据参数决定是创建一个新的检查点还是加载一个已存在的检查点。
    if not args.resume:
    # 如果 args.resume 参数的值为 False，则执行条件语句内的代码块。
    # args.resume 是一个布尔值，用于指示是否需要从先前的检查点恢复训练。
        checkpoint = {}
        # 初始化一个空字典 checkpoint，这个字典将用于存储训练过程中的重要信息，以便在需要时可以恢复训练。
        checkpoint['args'] = args
        # 将当前的参数 args 存储到 checkpoint 字典中，这样在训练过程中可以随时访问这些参数。
    else:
        # 如果 args.resume 参数的值为 True，则执行 else 代码块，即从检查点恢复训练。
        checkpoint = torch.load(f'save/{args.name}')
        # 使用 torch.load 函数加载之前保存的检查点文件。检查点文件的名称由 args.name 指定，通常包含训练配置的名称或标识。
        rounds = args.rounds
        # 从加载的检查点中获取 rounds 变量，它表示训练已经完成的轮次。
        iters = args.iters
        # 从加载的检查点中获取 iters 变量，它表示训练已经完成的迭代次数。
        device =args.device
        # 从加载的检查点中获取 device 变量，它表示训练使用的设备（GPU或CPU）。
        args = checkpoint['args']
        # 从检查点中获取并更新 args 参数，这将覆盖在脚本开始时解析的参数，确保训练过程使用检查点中的参数。
        args.resume = True
        # 更新 args.resume 参数为 True，以反映训练是从检查点恢复的。
        args.rounds = rounds
        # 更新 args.rounds 参数为从检查点获取的 rounds 值，以便继续训练。
        args.iters = iters
        # 更新 args.iters 参数为从检查点获取的 iters 值，以便继续训练。
        args.device = device
        # 更新 args.device 参数为从检查点获取的 device 值，确保训练在正确的设备上继续进行。
    ## Initialize RNGs and ensure reproducibility
    if args.seed is not None:
        # 如果命令行参数 args.seed 被设置了（即不是 None），则执行以下代码块。
        environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # 设置环境变量 CUBLAS_WORKSPACE_CONFIG，这个变量是用于控制CUDA的CUBLAS库的行为。
        # 这里设置的值 :4096:8 指定了CUBLAS使用的空间配置，以确保在不同运行之间保持一致性。
        torch.backends.cudnn.benchmark = False
        # 设置PyTorch后端的 cudnn.benchmark 为 False，禁用cuDNN的基准测试模式。
        # 基准测试模式可能会根据输入数据的特定属性改变算法，这可能导致结果不可复现。
        torch.use_deterministic_algorithms(True)
        # 启用PyTorch的确定性算法模式，确保使用确定性的算法，增加实验的可复现性。
        if not args.resume:
            # 如果命令行参数 args.resume 为 False（即不是从检查点恢复训练），则执行以下代码块。
            torch.manual_seed(args.seed)
            # 设置PyTorch的随机数生成器的种子为 args.seed。
            np.random.seed(args.seed)
            # 设置NumPy的随机数生成器的种子为 args.seed。
            random.seed(args.seed)
            # 设置Python标准库 random 模块的随机数生成器的种子为 args.seed。
        else:
            # 如果命令行参数 args.resume 为 True（即从检查点恢复训练），则执行以下代码块。
            torch.set_rng_state(checkpoint['torch_rng_state'])
            # 从检查点中恢复PyTorch的随机数生成器状态。
            np.random.set_state(checkpoint['numpy_rng_state'])
            # 从检查点中恢复NumPy的随机数生成器状态。
            random.setstate(checkpoint['python_rng_state'])
            # 从检查点中恢复Python标准库 random 模块的随机数生成器状态。
    # Load datasets and splits
    if not args.resume:
        # 如果 args.resume 参数为 False（即不是从检查点恢复训练），则执行以下代码块。
        datasets = getattr(datasets, args.dataset)(args, args.dataset_args)
        # 根据 args.dataset 参数的值，动态获取 datasets 模块中相应的数据集类，并使用该类创建数据集实例。
        # args.dataset_args 可能包含一些额外的参数，用于初始化数据集。
        splits = get_splits(datasets, args.num_clients, args.iid, args.balance)
        # 调用 get_splits 函数，根据数据集、客户端数量、是否独立同分布（I.I.D.）和平衡性要求来生成数据拆分。
        datasets_actual = {}
        # 初始化一个空字典 datasets_actual，用于存储实际使用的数据集。
        for dataset_type in splits:
            # 遍历 splits 字典中的每个数据集类型（如训练集、验证集、测试集）。
            if splits[dataset_type] is not None:
                # 如果当前数据集类型在 splits 中有定义（即不是 None），则执行以下代码块。
                idxs = []
                for client_id in splits[dataset_type].idxs:
                    idxs += splits[dataset_type].idxs[client_id]
                # 初始化一个空列表 idxs，用于存储当前数据集类型中所有客户端的样本索引。
                datasets_actual[dataset_type] = Subset(datasets[dataset_type], idxs)
                # 使用 Subset 类创建一个新的数据集，包含当前数据集类型中的样本索引 idxs。
            else:
                datasets_actual[dataset_type] = None
                # 如果当前数据集类型在 splits 中没有定义，则将 datasets_actual[dataset_type] 设置为 None。
        checkpoint['splits'] = splits
        # 将数据拆分信息 splits 存储到检查点字典中。
        checkpoint['datasets_actual'] = datasets_actual
        # 将实际使用的数据集信息 datasets_actual 存储到检查点字典中。
    else:
        # 如果 args.resume 参数为 True（即从检查点恢复训练），则执行以下代码块。
        splits = checkpoint['splits']
        # 从检查点中恢复数据拆分信息。
        datasets_actual = checkpoint['datasets_actual']
        # 从检查点中恢复实际使用的数据集信息。
    acc_types = ['train', 'test'] if datasets_actual['valid'] is None else ['train', 'valid']
    # 根据 datasets_actual 中是否有验证集（valid），设置 acc_types 列表。
    # 如果不存在验证集，则包含训练集和测试集；如果存在验证集，则包含训练集和验证集。
    # Load model
    # 这一行是注释，说明接下来的代码用于加载模型。
    num_classes = len(datasets_actual['train'].classes)
    # 获取训练数据集的类别数，这通常用于初始化模型的输出层。
    num_channels = datasets_actual['train'][0][0].shape[0]
    # 获取训练数据集中第一个图像样本的通道数，这通常用于初始化模型的输入层。
    model = getattr(models, args.model)(num_classes, num_channels, args.model_args).to(args.device)
    # 根据 args.model 参数的值，动态获取 models 模块中相应的模型类，并使用该类创建模型实例。
    # args.model_args 可能包含一些额外的参数，用于初始化模型。然后将模型移动到指定的设备（GPU或CPU）。
    if args.resume:
        model.load_state_dict(checkpoint['model_state_dict'])
    # 如果从检查点恢复训练，则从检查点中加载模型的状态字典。
    # Load optimizer and scheduler 加载优化器和调度器。
    optim = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    # 根据 args.optim 参数的值，动态获取 optimizers 模块中相应的优化器类，并使用该类创建优化器实例。
    # model.parameters() 提供模型的参数，args.optim_args 可能包含一些额外的参数，用于初始化优化器。
    sched = getattr(schedulers, args.sched)(optim, args.sched_args)
    # 根据 args.sched 参数的值，动态获取 schedulers 模块中相应的调度器类，并使用该类创建调度器实例。
    # optim 是优化器实例，args.sched_args 可能包含一些额外的参数，用于初始化调度器。
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        sched.load_state_dict(checkpoint['sched_state_dict'])
    # 如果从检查点恢复训练，则从检查点中加载优化器和调度器的状态字典。
    # Create clients  创建客户端
    if not args.resume:
        # 如果不是从检查点恢复训练，则执行以下代码块。
        clients = []
        # 初始化一个空列表 clients，用于存储客户端实例。
        for client_id in range(args.num_clients):
            # 遍历从 0 到 args.num_clients-1 的客户端编号。
            client_idxs = {dataset_type: splits[dataset_type].idxs[client_id] if splits[dataset_type] is not None else None for dataset_type in splits}
            # 为每个客户端创建一个索引字典 client_idxs，包含客户端在每个数据集类型中的样本索引。
            clients.append(Client(args=args, datasets=datasets, idxs=client_idxs))
            # 创建一个 Client 实例，使用 args 参数、数据集实例 datasets 和客户端索引 client_idxs，并将其添加到 clients 列表中。
        checkpoint['clients'] = clients
        # 将客户端列表存储到检查点字典中。
    else:
        clients = checkpoint['clients']
        # 如果从检查点恢复训练，则从检查点中恢复客户端列表。
    # Set client sampling probabilities 设置客户端采样概率
    if args.vc_size is not None:
        # 如果 args.vc_size 参数被设置了（即不是 None），则执行以下代码块。vc_size 可能表示每个客户端的数据量。
        # Proportional to the number of examples (FedVC)
        p_clients = np.array([len(client.loaders['train'].dataset) for client in clients])
        # 计算每个客户端训练数据集的大小，并将其存储在一个 NumPy 数组 p_clients 中。
        # 这通常用于实现按数据量比例采样（Federated Virtual Cluster，FedVC）。
        p_clients = p_clients / p_clients.sum()
        # 将 p_clients 数组中的每个元素除以其总和，从而得到每个客户端的采样概率，这些概率与其数据量成比例。
    else:
        # Uniform
        p_clients = None
        # 如果 args.vc_size 参数没有被设置，则不使用按数据量比例采样，而是可能使用均匀采样或其他采样策略，此时将 p_clients 设置为 None。
    # Determine number of clients to sample per round 确定每轮采样的客户端数量
    m = max(int(args.frac_clients * args.num_clients), 1)
    # 根据参数 args.frac_clients（每轮采样的客户端比例）和 args.num_clients（总客户端数量），计算每轮要采样的客户端数量 m。
    # 使用 max 函数确保 m 至少为 1，即每轮至少采样一个客户端。
    # Print experiment summary 打印实验摘要
    summary = exp_details(args, model, datasets_actual, splits)
    # 调用 exp_details 函数生成实验摘要，摘要中可能包含参数、模型、数据集和数据拆分的详细信息。
    print('\n' + summary)
    # 打印实验摘要，换行符 \n 确保摘要从新行开始打印。
    # Log experiment summary, client distributions, example images 记录实验摘要、客户端分布和示例图像
    if not args.no_log:
        # 如果 args.no_log 参数为 False（即没有指定不记录日志），则执行以下代码块。
        logger = SummaryWriter(f'runs/{args.name}')
        # 创建一个 SummaryWriter 实例，它将用于将信息写入TensorBoard。
        # 日志将保存在 runs/{args.name} 目录下，其中 args.name 是实验的名称或标识。
        if not args.resume:
            # 如果 args.resume 参数为 False（即不是从检查点恢复训练），则执行以下代码块。
            logger.add_text('Experiment summary', re.sub('^', '    ', re.sub('\n', '\n    ', summary)))
            # 将实验摘要 summary 添加到TensorBoard的文本日志中。使用正则表达式 re.sub 确保文本的格式正确。
            splits_fig = get_splits_fig(splits, args.iid, args.balance)
            # 调用 get_splits_fig 函数获取数据拆分的图形表示，这可能包括训练集、验证集和测试集的分布。
            logger.add_figure('Splits', splits_fig)
            # 将数据拆分的图形表示添加到TensorBoard。
            datasets_fig = get_datasets_fig(datasets_actual, args.train_bs)
            # 调用 get_datasets_fig 函数获取数据集的图形表示，这可能包括数据集中样本的分布或样本图像。
            logger.add_figure('Datasets', datasets_fig)
            # 将数据集的图形表示添加到TensorBoard。
            input_size = (1,) + tuple(datasets_actual['train'][0][0].shape)
            # 根据训练数据集中第一个样本的大小，确定模型输入的尺寸。
            fake_input = torch.zeros(input_size).to(args.device)
            # 创建一个与模型输入尺寸相同的零张量 fake_input，并将其移动到指定的设备（GPU或CPU）。
            logger.add_graph(model, fake_input)
            # 将模型的计算图和 fake_input 添加到TensorBoard，这有助于可视化模型的结构。
    else:
        logger = None
        # 如果 args.no_log 参数为 True（即指定不记录日志），则不创建 SummaryWriter 实例，logger 设置为 None。
    if not args.resume:
        # Compute initial average accuracies
        # 如果 args.resume 参数为 False，即训练不是从先前的状态恢复。
        acc_avg = get_acc_avg(acc_types, clients, model, args.device)
        # 调用 get_acc_avg 函数计算初始的平均准确率。
        # 这个函数可能接受客户端列表、模型、设备等参数，并返回不同数据集（例如训练集和测试集）上的平均准确率。
        acc_avg_best = acc_avg[acc_types[1]]
        # 从计算出的初始平均准确率中获取 acc_types 列表第二个元素（通常是测试集）的准确率，并将其存储在 acc_avg_best 中，用于跟踪最佳性能。
        # Print and log initial stats
        if not args.quiet:
            # 如果 args.quiet 参数为 False，即用户希望看到详细的打印输出。
            print('Training:')
            print('    Round: 0' + (f'/{args.rounds}' if args.iters is None else ''))
            # 打印训练开始的信息，包括当前轮数（0）和总轮数（如果 args.iters 为 None）。
        loss_avg, lr = torch.nan, torch.nan
        # 初始化平均损失 loss_avg 和学习率 lr 为 torch.nan，表示在第一轮之前没有可用的损失和学习率数据。
        printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, 0, 0, args.iters)
        # 调用 printlog_stats 函数来打印和记录初始统计信息，包括是否安静模式、日志记录器、平均损失、平均准确率、准确率类型、学习率、当前轮数、当前迭代次数和总迭代次数。
    else:
        acc_avg_best = checkpoint['acc_avg_best']
        # 如果训练是从检查点恢复的（即 args.resume 为 True），则从检查点字典中恢复 acc_avg_best。

    init_end_time = time()
    # 记录初始化阶段结束的时间，这个时间戳可以用来计算初始化阶段所花费的时间。
    # Train server model 训练服务器模型
    if not args.resume:
        # 如果 args.resume 参数为 False，即训练不是从先前的状态恢复。
        last_round = -1
        # 初始化变量 last_round 为 -1，表示还没有开始训练任何轮次。
        iter = 0
        # 初始化迭代次数 iter 为 0，表示还没有进行任何迭代。
        v = None
        # 初始化变量 v 为 None，这个变量可能用于累积更新或动量项。
    else:
        # 如果 args.resume 参数为 True，即训练是从检查点恢复的。
        last_round = checkpoint['last_round']
        # 从检查点中恢复 last_round，即训练已经完成的最后轮次。
        iter = checkpoint['iter']
        # 从检查点中恢复 iter，即训练已经完成的迭代次数。
        v = checkpoint['v']
        # 从检查点中恢复变量 v，这个变量可能存储了之前训练中的累积更新或动量项。
    for round in range(last_round + 1, args.rounds):
        # 开始一个循环，从上一轮次 last_round + 1 开始，到参数 args.rounds 指定的轮次数结束。
        if not args.quiet:
            print(f'    Round: {round+1}' + (f'/{args.rounds}' if args.iters is None else ''))
        # 如果不在安静模式下，打印当前的训练轮次。
        # Sample clients 采样客户端
        client_ids = np.random.choice(range(args.num_clients), m, replace=False, p=p_clients)
        # 根据之前设置的采样概率 p_clients 从所有客户端中随机选择 m 个客户端
        # Train client models 训练客户端模型
        updates, num_examples, max_iters, loss_tot = [], [], 0, 0.
        # 初始化用于记录客户端更新、客户端样本数量、最大迭代次数和总损失的变量
        for i, client_id in enumerate(client_ids):
            # 遍历所有被采样的客户端
            if not args.quiet: print(f'        Client: {client_id} ({i+1}/{m})')
            # 如果不在安静模式下，打印正在训练的客户端编号
            client_model = deepcopy(model)
            # 对服务器模型进行深拷贝，以便每个客户端可以有自己的模型副本进行训练。
            optim.__setstate__({'state': defaultdict(dict)})
            # 重置优化器的状态。
            optim.param_groups[0]['params'] = list(client_model.parameters())
            # 将优化器的参数组更新为客户端模型的参数。
            client_update, client_num_examples, client_num_iters, client_loss = clients[client_id].train(model=client_model, optim=optim, device=args.device)
            # 调用客户端的 train 方法进行训练，获取客户端模型的更新、客户端的样本数量、迭代次数和损失。
            if client_num_iters > max_iters: max_iters = client_num_iters
            # 更新最大迭代次数。
            if client_update is not None:
                updates.append(deepcopy(client_update))
                loss_tot += client_loss * client_num_examples
                num_examples.append(client_num_examples)
            # 如果客户端返回了更新，则将其添加到更新列表中，并更新总损失和样本数量
        iter += max_iters
        # 更新全局迭代次数
        lr = optim.param_groups[0]['lr']
        # 获取当前的学习率
        if len(updates) > 0:
            # Update server model
            # 如果存在客户端更新，执行服务器模型更新。
            update_avg = average_updates(updates, num_examples)

            if v is None:
                v = deepcopy(update_avg)
            else:
                for key in v.keys():
                    v[key] = update_avg[key] + v[key] * args.server_momentum
            #new_weights = deepcopy(model.state_dict())
            #for key in new_weights.keys():
            #new_weights[key] = new_weights[key] - v[key] * args.server_lr
            #model.load_state_dict(new_weights)
            for key in model.state_dict():
                model.state_dict()[key] -= v[key] * args.server_lr

            # Compute round average loss and accuracies
            if round % args.server_stats_every == 0:
                loss_avg = loss_tot / sum(num_examples)
                acc_avg = get_acc_avg(acc_types, clients, model, args.device)

                if acc_avg[acc_types[1]] > acc_avg_best:
                    acc_avg_best = acc_avg[acc_types[1]]

        # Save checkpoint
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['optim_state_dict'] = optim.state_dict()
        checkpoint['sched_state_dict'] = sched.state_dict()
        checkpoint['last_round'] = round
        checkpoint['iter'] = iter
        checkpoint['v'] = v
        checkpoint['acc_avg_best'] = acc_avg_best
        checkpoint['torch_rng_state'] = torch.get_rng_state()
        checkpoint['numpy_rng_state'] = np.random.get_state()
        checkpoint['python_rng_state'] = random.getstate()
        torch.save(checkpoint, f'save/{args.name}')

        # Print and log round stats
        if round % args.server_stats_every == 0:
            printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, round+1, iter, args.iters)

        # Stop training if the desired number of iterations has been reached
        if args.iters is not None and iter >= args.iters: break

        # Step scheduler
        if type(sched) == schedulers.plateau_loss:
            sched.step(loss_avg)
        else:
            sched.step()

    train_end_time = time()

    # Compute final average test accuracy
    acc_avg = get_acc_avg(['test'], clients, model, args.device)

    test_end_time = time()

    # Print and log test results
    print('\nResults:')
    print(f'    Average test accuracy: {acc_avg["test"]:.3%}')
    print(f'    Train time: {timedelta(seconds=int(train_end_time-init_end_time))}')
    print(f'    Total time: {timedelta(seconds=int(time()-start_time))}')

    if logger is not None: logger.close()
