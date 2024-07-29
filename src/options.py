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

import argparse
# 导入Python标准库中的argparse模块，该模块用于创建命令行接口，解析命令行参数。
from inspect import getmembers, isfunction, isclass
# 从inspect模块导入getmembers、isfunction和isclass函数。这些函数用于获取对象的成员、检查成员是否是函数以及是否是类。
from ast import literal_eval
# 从ast模块导入literal_eval函数，该函数用于将字符串形式的Python字面量（如字符串、数字、元组等）安全地转换为对应的Python对象。
from datetime import datetime
# 从 datetime 模块导入 datetime 类，该类用于处理日期和时间。
import sys
# 导入Python标准库中的sys模块，该模块提供了与Python解释器和它的环境交互的接口。
from torch.cuda import device_count
# 从PyTorch的torch.cuda模块导入device_count函数，该函数用于获取系统中可用的CUDA设备数量。
import datasets, models, optimizers, schedulers
# 导入自定义模块 datasets、models、optimizers 和 schedulers。这些模块可能包含数据集加载、模型定义、优化器和学习率调度器的实现。


def args_parser():
    # 定义了 args_parser 函数。
    #max_help_position=1000, width=1000
    usage = 'python main.py [ARGUMENTS]'
    parser = argparse.ArgumentParser(prog='main.py', usage=usage, add_help=False, formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog))
    # 设置了命令行工具的使用说明和参数解析器的配置。
    # Algorithm arguments
    args_algo = parser.add_argument_group('algorithm arguments')
    # 添加了一个名为“algorithm arguments”的参数分组。
    args_algo_rounds_iters = args_algo.add_mutually_exclusive_group()
    #   --rounds ROUNDS
    #   number of communication rounds, or number of epochs if --centralized (default: 200)
    args_algo_rounds_iters.add_argument('--rounds', type=int, default=200,
                        help="number of communication rounds, or number of epochs if --centralized")
    #   --iters ITERS
    #   number of iterations: the iterations of a round are determined by the client with the
    #   largest number of images (default: None)
    args_algo_rounds_iters.add_argument('--iters', type=int, default=None,
                        help="number of iterations: the iterations of a round are determined by the client with the largest number of images")
    args_algo.add_argument('--num_clients', '-K', type=int, default=100,
                        help="number of clients")
    #   --num_clients
    #   NUM_CLIENTS, -K NUM_CLIENTS number of clients (default: 100)
    args_algo.add_argument('--frac_clients', '-C', type=float, default=0.1,
                        help="fraction of clients selected at each round")
    #   --frac_clients
    #   FRAC_CLIENTS, -C FRAC_CLIENTS fraction of clients selected at each round (default: 0.1)
    args_algo.add_argument('--train_bs', '-B', type=int, default=50,
                        help="client training batch size, 0 to use the whole training set")
    #   --train_bs
    #   TRAIN_BS, -B TRAIN_BS client training batch size, 0 to use the whole training set (default: 50)
    args_algo.add_argument('--epochs', '-E', type=int, default=5,
                        help="number of client epochs")
    #   --epochs
    #   EPOCHS, -E EPOCHS number of client epochs (default: 5)
    args_algo.add_argument('--hetero', type=float, default=0,
                        help="probability of clients being stragglers, i.e. training for less than EPOCHS epochs")
    #   --hetero HETERO
    #   probability of clients being stragglers, i.e. training for less than EPOCHS epochs (default: 0)
    args_algo.add_argument('--drop_stragglers', action='store_true', default=False,
                        help="drop stragglers")
    #   --drop_stragglers
    #   drop stragglers (default: False)
    args_algo.add_argument('--server_lr', type=float, default=1,
                        help="server learning rate")
    # #   --server_lr
    # SERVER_LR server learning rate (default: 1)
    args_algo.add_argument('--server_momentum', type=float, default=0,
                        help="server momentum for FedAvgM algorithm")
    #   --server_momentum
    #   SERVER_MOMENTUM server momentum for FedAvgM algorithm (default: 0)
    args_algo.add_argument('--mu', type=float, default=0,
                        help="mu parameter for FedProx algorithm")
    #   --mu MU
    #   mu parameter for FedProx algorithm (default: 0)
    args_algo.add_argument('--centralized', action='store_true', default=False,
                        help="use centralized algorithm")
    #   --centralized
    #   use centralized algorithm (default: False)
    args_algo.add_argument('--fedsgd', action='store_true', default=False,
                        help="use FedSGD algorithm")
    #   --fedsgd
    #   use FedSGD algorithm (default: False)
    args_algo.add_argument('--fedir', action='store_true', default=False,
                        help="use FedIR algorithm")
    #   --fedir
    #   use FedIR algorithm (default: False)
    args_algo.add_argument('--vc_size', type=int, default=None,
                        help="use FedVC algorithm with virtual client size VC_SIZE")
    #   --vc_size
    #   VC_SIZE     use FedVC algorithm with virtual client size VC_SIZE (default: None)
    # Dataset and split arguments
    args_dataset_split = parser.add_argument_group('dataset and split arguments')
    args_dataset_split.add_argument('--dataset', type=str, default='cifar10', choices=[f[0] for f in getmembers(datasets, isfunction) if f[1].__module__ == 'datasets'],
                        help="dataset, place yours in datasets.py")
    #   --dataset
    #   {cifar10,fmnist,mnist} dataset, place yours in datasets.py (default: cifar10)
    args_dataset_split.add_argument('--dataset_args', type=str, default='augment=True',
                        help="dataset arguments")
    #   --dataset_args
    #   DATASET_ARGS dataset arguments (default: augment=True)
    args_dataset_split.add_argument('--frac_valid', type=float, default=0,
                        help="fraction of the training set to use for validation")
    #   --frac_valid
    #   FRAC_VALID fraction of the training set to use for validation(default: 0)
    args_dataset_split.add_argument('--iid', type=float, default='inf',
                        help="identicalness of client distributions")
    #   --iid IID
    #   identicalness of client distributions (default: inf)
    args_dataset_split.add_argument('--balance', type=float, default='inf',
                        help="balance of client distributions")
    #   --balance
    #   BALANCE balance of client distributions (default: inf) Model, optimizer and scheduler arguments
    args_model_optim_sched = parser.add_argument_group('model, optimizer and scheduler arguments')
    args_model_optim_sched.add_argument('--model', type=str, default='mobilenet_v3', choices=[c[0] for c in getmembers(models, isclass) if c[1].__module__ == 'models'],
                        help="model, place yours in models.py")
    #   --model
    #   {cnn_cifar10,cnn_mnist,efficientnet,ghostnet,lenet5,lenet5_orig,mlp_mnist,mnasnet,mobilenet_v3} model,
    #   place yours in models.py (default: lenet5)
    args_model_optim_sched.add_argument('--model_args', type=str, default='ghost=True,norm=None',
                        help="model arguments")
    #   --model_args
    #   MODEL_ARGS model arguments (default: ghost=True,norm=None)
    args_model_optim_sched.add_argument('--optim', type=str, default='sgd', choices=[f[0] for f in getmembers(optimizers, isfunction)],
                        help="optimizer, place yours in optimizers.py")
    #   --optim {adam,sgd}
    #   optimizer, place yours in optimizers.py (default: sgd)
    args_model_optim_sched.add_argument('--optim_args', type=str, default='lr=0.01,momentum=0,weight_decay=4e-4',
                        help="optimizer arguments")
    #   --optim_args
    #   OPTIM_ARGS optimizer arguments (default: lr=0.01,momentum=0,weight_decay=4e-4)
    args_model_optim_sched.add_argument('--sched', type=str, default='fixed', choices=[c[0] for c in getmembers(schedulers, isclass) if c[1].__module__ == 'schedulers'],
                        help="scheduler, place yours in schedulers.py")
    #   --sched
    #   {const,fixed,plateau_loss,step} scheduler, place yours in schedulers.py (default: fixed)
    args_model_optim_sched.add_argument('--sched_args', type=str, default=None,
                        help="scheduler arguments")
    #   --sched_args
    #   SCHED_ARGS scheduler arguments (default: None)
    # Output arguments
    args_output = parser.add_argument_group('output arguments')
    args_output.add_argument('--client_stats_every', type=int, default=0,
                        help="compute and print client statistics every CLIENT_STATS_EVERY batches, 0 for every epoch")
    #   --client_stats_every
    #   CLIENT_STATS_EVERY compute and print client statistics every CLIENT_STATS_EVERY batches,
    #   0 for every epoch (default: 0)
    args_output.add_argument('--server_stats_every', type=int, default=1,
                        help="compute, print and log server statistics every SERVER_STATS_EVERY rounds")
    #   --server_stats_every
    #   SERVER_STATS_EVERY compute, print and log server statistics every SERVER_STATS_EVERY rounds (default: 1)
    args_output.add_argument('--name', type=str, default=None,
                        help="log to runs/NAME and save checkpoints to save/NAME, None for YYYY-MM-DD_HH-MM-SS")
    #   --name NAME
    #   log to runs/NAME and save checkpoints to save/NAME, None for YYYY-MM-DD_HH-MM-SS (default: None)
    args_output.add_argument('--no_log', action='store_true', default=False,
                        help="disable logging")
    #   --no_log
    #   disable logging (default: False)
    args_output.add_argument('--no_save', action='store_true', default=False,
                        help="disable checkpoints")
    #  --no_save
    #  disable checkpoints (default: False)
    args_output.add_argument('--quiet', '-q', action='store_true', default=False,
                        help="less verbose output")
    #   --quiet
    #   -q  less verbose output (default: False)
    # Other arguments
    args_other = parser.add_argument_group('other arguments')
    args_other.add_argument('--test_bs', type=int, default=256,
                        help="client test/validation batch size")
    #   --test_bs
    #   TEST_BS client test/validation batch size (default: 256)
    args_other.add_argument('--seed', type=int, default=0,
                        help="random seed")
    #   --seed
    #   SEED random seed (default: 0)
    args_other.add_argument('--device', type=str, default='cuda:0', choices=['cuda:%d' % device for device in range(device_count())] + ['cpu'],
                        help="device to train/validate/test with")
    #   --device
    #   {cuda:0,cpu} device to train/validate/test with (default: cuda:0)
    args_other.add_argument('--resume', action='store_true', default=False,
                        help="resume experiment from save/NAME checkpoint")
    #   --resume
    #   resume experiment from save/NAME checkpoint (default: False)
    args_other.add_argument('--help', '-h', action='store_true', default=False,
                        help="show this help message and exit")
    #   --help
    #   -h show this help message and exit (default: False)
    args = parser.parse_args()
    # 通过 argparse 的 parse_args() 方法解析命令行输入的参数，并将解析后的参数赋值给变量 args。
    # 这些参数被存储在一个命名空间中，可以通过属性访问。
    if args.help:
        parser.print_help()
        exit()
    # 如果用户通过命令行请求帮助（即设置了 args.help），则打印出帮助信息并退出程序。
    if args.iters is not None:
        args.rounds = sys.maxsize
    # 如果用户指定了迭代次数（args.iters），则将通信轮数（args.rounds）设置为 sys.maxsize，
    # 这通常意味着无限轮数，直到达到用户指定的迭代次数。
    if args.name is None:
        args.name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 如果用户没有指定实验的名称（args.name），则使用当前日期和时间生成一个名称。
    args.dataset_args = args_str_to_dict(args.dataset_args)
    # 将 args.dataset_args 参数（如果提供）从字符串转换为字典。这个转换函数 args_str_to_dict 需要在代码其他地方定义。
    args.model_args = args_str_to_dict(args.model_args)
    # 对模型参数进行相同的字符串到字典的转换。
    args.optim_args = args_str_to_dict(args.optim_args)
    # 对优化器参数进行字符串到字典的转换。
    args.sched_args = args_str_to_dict(args.sched_args)
    # 对调度器参数进行字符串到字典的转换。
    if args.vc_size is not None:
        args.epochs = 1
    # 如果参数 args.vc_size 有值（即不是 None），则将训练轮数 args.epochs 设置为 1。
    # 这通常意味着使用虚拟客户端（Virtual Client，FedVC）策略时，每个客户端只进行一个 epoch 的训练。
    if args.fedsgd:
        args.epochs = 1
        args.train_bs = 0
    # 如果参数 args.fedsgd 为 True，表明使用联邦随机梯度下降（FedSGD）算法。
    # 在这种情况下，将训练轮数 args.epochs 设置为 1，并将训练批量大小 args.train_bs 设置为 0，意味着每个客户端在整个数据集上进行训练。
    if args.centralized:
        # 如果参数 args.centralized 为 True，表明使用集中式算法。在这种情况下：
        args.num_clients = 1
        # args.num_clients 设置为 1，表示只有一个客户端。
        args.frac_clients = 1
        # args.frac_clients 设置为 1，表示在每轮中使用所有客户端。
        args.epochs = 1
        # args.epochs 设置为 1，表示进行一个训练轮次。
        args.hetero = 0
        # args.hetero 设置为 0，表示客户端的异构性为零（所有客户端相同）。
        args.iid = float('inf')
        # args.iid 设置为正无穷，表示客户端数据分布的一致性为正无穷（完全相同的数据）。
        args.balance = float('inf')
        # args.balance 设置为正无穷，表示客户端数据分布的平衡性为正无穷（完全平衡）。
        args.vc_size = None
        # args.vc_size 设置为 None，表示不使用虚拟客户端。
        args.fedir = False
        # args.fedir 设置为 False，表示不使用联邦重要性重加权（FedIR）。
        args.mu = 0
        # args.mu 设置为 0，表示不使用 FedProx 算法的 mu 参数。
        args.fedsgd = False
        # args.fedsgd 设置为 False，表示不使用 FedSGD 算法。
        args.server_lr = 1
        # args.server_lr 设置为 1，表示设置服务器学习率为 1。
        args.server_momentum = 0
        # args.server_momentum 设置为 0，表示设置服务器动量为 0。
    return args


# 这个函数通过解析一个格式化的字符串（其中参数以逗号分隔，键值对以等号分隔），
# 将其转换为一个字典，使得参数可以方便地用于程序中的配置。
# 例如，字符串 "lr=0.01,momentum=0.9" 将被转换为 {'lr': 0.01, 'momentum': 0.9} 这样的字典。
# 这种格式的字符串通常在命令行参数中使用，方便用户一次性传入多个参数。
def args_str_to_dict(args_str):
    # 这段代码定义了一个名为 args_str_to_dict 的函数，其作用是将一个包含参数的字符串转换为一个字典。
    # 这个函数对于解析命令行参数中的参数字符串非常有用。以下是对函数中每一行代码的详细解释：
    # 定义了一个名为 args_str_to_dict 的函数，它接收一个参数 args_str，这个参数是一个包含参数的字符串。
    args_dict = {}
    # 初始化一个空字典 args_dict，用于存储解析后的参数。
    if args_str is not None:
        # 检查传入的字符串 args_str 是否不是 None。
        for arg in args_str.replace(' ', '').split(','):
            # 遍历由逗号分隔的参数字符串列表。首先，使用 replace(' ', '') 移除所有空格，然后使用 split(',') 按逗号分割字符串。
            keyvalue = arg.split('=')
            # 对于每个参数，使用 split('=') 分割键和值。
            args_dict[keyvalue[0]] = literal_eval(keyvalue[1])
            # 使用 literal_eval 函数将值从字符串转换为 Python 的字面量数据类型（如整数、浮点数、字符串、元组等），
            # 并将键值对存入 args_dict 字典中。
    return args_dict
    # 返回填充好的 args_dict 字典。


