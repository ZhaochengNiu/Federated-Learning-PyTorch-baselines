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


class Split():
    def __init__(self, idxs, dist, emd):
        # Split 类的构造函数接收三个参数：
        # idxs：一个数组或列表，包含数据点的索引。
        # dist：一个分布矩阵，可能表示数据点在某些特征上的分布。
        # emd：一个数值，表示 Earth Mover's Distance。
        self.idxs = idxs
        # 将传入的 idxs 赋值给实例变量 self.idxs。
        self.dist = dist
        # 将传入的 dist 赋值给实例变量 self.dist。
        self.emd = emd
        # 将传入的 emd 赋值给实例变量 self.emd。


def earthmover_distance(dist):
    dist = dist[~torch.all(dist == 0, axis=1)]
    # 移除 dist 矩阵中所有行元素都为0的行。这里使用了 torch.all 函数检查每一行是否全为0，然后取反（~）来选择非全0的行。
    N_client = dist.sum(1, keepdims=True)
    # 计算每个客户端的数据点总数。sum(1) 表示对每一行求和，keepdims=True 确保结果仍然是二维的，即使只有一个元素。
    N = dist.sum()
    # 计算所有客户端的数据点总数。
    q = dist / N_client
    # 计算每个客户端的数据点分布，即每个客户端的分布占其总数的比例。
    # 这里通过将 dist 除以 N_client 来归一化每个客户端的分布。
    p = (dist).sum(0, keepdims=True) / N
    # 计算全局数据点分布，即每个类别的分布占总数的比例。sum(0) 表示对每一列求和，然后除以总数据点数 N。
    emd = (torch.abs(q - p).sum(1, keepdims=True) * N_client).sum() / N
    # 计算 EMD。首先计算每个客户端分布 q 与全局分布 p 之间的绝对差异，然后对这些差异求和（sum(1)），得到每个客户端的 EMD 贡献。
    # 将这些贡献乘以每个客户端的数据点总数 N_client，最后将所有客户端的贡献求和并除以总数据点数 N，得到最终的 EMD。
    return emd
    # 返回计算得到的 EMD 值


def get_split(dataset, q_class, q_client):
    if dataset is None:
        return None

    num_clients, num_classes = q_class.shape

    '''
    dist = (q_class*(q_client*len(dataset)).to(int)).to(int)

    if no_replace:
        num_class_examples = torch.tensor([(np.array(dataset.targets) == cls).sum() for cls in range(num_classes)])
        if (dist.sum(0) > num_class_examples).any():
            raise ValueError('Invalid --iid and/or --balance for --no_replace')

    split = {}
    for cls in range(num_classes):
        idxs_class = set((np.array(dataset.targets) == cls).nonzero()[0])
        for client_id in range(num_clients):
            if cls == 0: split[client_id] = []
            idxs_class_client = list(np.random.choice(list(idxs_class), dist[client_id,cls].item(), replace=not no_replace))
            split[client_id] += idxs_class_client
            if no_replace:
                idxs_class = idxs_class - set(idxs_class_client)
    '''

    split_idxs = {client_id: [] for client_id in range(num_clients)}
    q_class_tilde = deepcopy(q_class)
    split_dist = torch.zeros(num_clients, num_classes)

    num_images_clients = (q_client * len(dataset)).round().to(int)
    delta_images = len(dataset) - num_images_clients.sum().item()
    client_id = 0
    for i in range(abs(delta_images)):
        num_images_clients[client_id % num_clients] += np.sign(delta_images)
        client_id += 1

    classes = set(range(num_classes))
    idxs_classes = [set((np.array(dataset.targets) == cls).nonzero()[0]) for cls in range(num_classes)]
    num_images = len(dataset)

    while(1):
        for cls in range(num_classes):
            if len(idxs_classes[cls]) > 0:
                for client_id in range(num_clients):
                    if num_images_clients[client_id] > 0:
                        num_images_client_class = min((q_class_tilde[client_id, cls] * num_images_clients[client_id]).round().to(int).item(), len(idxs_classes[cls]))
                        idxs_client_class = list(np.random.choice(list(idxs_classes[cls]), num_images_client_class, replace=False))
                        split_idxs[client_id] += idxs_client_class
                        idxs_classes[cls] -= set(idxs_client_class)
                        num_images_clients[client_id] -= num_images_client_class
                        split_dist[client_id, cls] += num_images_client_class
                        if len(idxs_classes[cls]) == 0 and len(classes) > 1:
                            classes -= {cls}
                            q_class_tilde[:, cls] = 0
                            idxs = (q_class_tilde == 0).all(1)
                            q_class_tilde[idxs, list(classes)[0]] = 1
                            q_class_tilde /= q_class_tilde.sum(1, keepdim=True)
                            break

        if num_images_clients.sum() == 0: break

    split_emd = {}
    split_emd['class'] = earthmover_distance(split_dist)
    split_emd['client'] = torch.abs(split_dist.sum(1)/split_dist.sum() - torch.tensor([1/num_clients]*num_clients)).sum()

    return Split(split_idxs, split_dist, split_emd)


def get_splits(datasets, num_clients, iid, balance):
    num_classes = len(datasets['train'].classes)

    if iid == 0:
        q_class = torch.zeros((num_clients, num_classes))
        for client_id in range(num_clients):
            q_class[client_id, np.random.randint(low=0, high=num_classes)] = 1
    elif iid == float('inf'):
        q_class = torch.tensor([(datasets['train'].targets == cls).sum() for cls in range(len(datasets['train'].classes))])
        q_class = q_class / len(datasets['train'])
        q_class = q_class.repeat(num_clients, 1)
    else:
        p_class = torch.tensor([(datasets['train'].targets == cls).sum() for cls in range(len(datasets['train'].classes))])
        p_class = p_class / len(datasets['train'])
        q_class = torch.distributions.dirichlet.Dirichlet(iid*p_class).sample((num_clients,))

    if balance == 0:
        q_client = torch.zeros(num_clients).reshape((num_clients,1))
        q_client[np.random.randint(low=0, high=num_clients)] = 1
    elif balance == float('inf'):
        q_client = (torch.ones(num_clients).divide(num_clients)).reshape((num_clients,1))
    else:
        p_client = torch.ones(num_clients).divide(num_clients)
        q_client = torch.distributions.dirichlet.Dirichlet(balance*p_client).sample().reshape((num_clients,1))

    splits = {}
    for key in datasets.keys():
        splits[key] = get_split(datasets[key], q_class, q_client)

    return splits


def get_splits_fig(splits, iid, balance):
    types, titles = [], []
    for type in splits:
        if splits[type] is not None:
            types.append(type)
            titles.append(type.capitalize())
    fig, ax = plt.subplots(1, len(types))

    iid_str = '∞' if iid == float('inf') else '%g' % iid
    balance_str = '∞' if balance == float('inf') else '%g' % balance

    num_clients, num_classes = splits['train'].dist.shape
    y = torch.arange(num_clients)
    for i, type in enumerate(types):
        left = torch.zeros(num_clients)
        for c in range(num_classes):
            ax[i].barh(y, splits[type].dist[:,c], left=left, height=1)
            left += splits[type].dist[:,c]
        ax[i].set_xlim((0,max(left)))
        ax[i].set_xlabel('Class distribution')
        ax[i].set_title(titles[i])
        if i == 0:
            ax[i].set_ylabel('Client')
        else:
            ax[i].set_yticks([])

    fig.suptitle('$α_{class} = %s, α_{client} = $%s' % (iid_str, balance_str))
    fig.tight_layout()
    fig.set_size_inches(4*len(types), 4)

    return fig
