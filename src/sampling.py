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


# Split 类可以用于表示数据集中的一个子集，其中 idxs 指定了子集数据点的索引，dist 描述了这些数据点的分布特征，
# 而 emd 提供了这个子集与整体数据集或其他子集之间差异的量化度量。这种类在处理数据拆分、分布分析和比较时非常有用，
# 尤其是在机器学习和数据挖掘领域。
class Split():
    def __init__(self, idxs, dist, emd):
        # 这段代码定义了一个名为 Split 的类，它用于封装与数据拆分相关的信息，
        # 如数据点索引、分布矩阵和 Earth Mover's Distance (EMD)。以下是对类及其构造函数的逐行解释：
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


# 这个 earthmover_distance 函数通过比较每个客户端的数据分布与全局数据分布之间的差异来计算 EMD，
# 这在评估数据分布的均匀性或在联邦学习中评估数据异质性时非常有用。函数返回的 EMD 值越小，表示客户端之间的数据分布越接近。
def earthmover_distance(dist):
    # 这段代码定义了一个名为 earthmover_distance 的函数，用于计算两个概率分布之间的 Earth Mover's Distance (EMD)，
    # 也称为 Wasserstein 距离。以下是对函数中每一行代码的详细解释：
    # 定义了一个名为 earthmover_distance 的函数，它接收一个参数 dist，这个参数是一个分布矩阵。
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


# 这个 get_split 函数实现了一个复杂的数据拆分逻辑，考虑了类别分布和客户端分布，以及图像分配的策略。
# 通过这种方式，可以生成不同数据集类型的拆分，这些拆分可以用于联邦学习或其他分布式学习场景。
def get_split(dataset, q_class, q_client):
    # 这段代码定义了一个名为 get_split 的函数，它用于根据给定的数据集、类别分布和客户端分布生成数据拆分。
    # 以下是对函数及其参数的详细解释：
    # 定义了一个名为 get_split 的函数，它接收以下参数：
    # dataset：要拆分的数据集。
    # q_class：类别分布矩阵。
    # q_client：客户端分布向量。
    if dataset is None:
        return None
    # 如果传入的数据集为 None，则直接返回 None。
    num_clients, num_classes = q_class.shape
    # 获取类别分布矩阵 q_class 的形状，即客户端数量和类别数量。
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
    # 初始化一个字典 split_idxs，用于存储每个客户端的样本索引。
    q_class_tilde = deepcopy(q_class)
    # 使用 deepcopy 函数复制 q_class 矩阵，得到 q_class_tilde。
    split_dist = torch.zeros(num_clients, num_classes)
    # 创建一个形状为 (num_clients, num_classes) 的零矩阵 split_dist，用于存储拆分分布。
    num_images_clients = (q_client * len(dataset)).round().to(int)
    # 计算每个客户端应分配的图像数量，并将结果四舍五入到最近的整数。
    delta_images = len(dataset) - num_images_clients.sum().item()
    # 计算图像总数与分配给客户端的图像总数之差。
    client_id = 0
    for i in range(abs(delta_images)):
        num_images_clients[client_id % num_clients] += np.sign(delta_images)
        client_id += 1
    # 调整 num_images_clients 以确保所有图像都被分配。
    # 如果 delta_images 大于 0，将剩余的图像分配给前几个客户端；如果小于 0，则从未分配的客户端中减去相应数量的图像。
    classes = set(range(num_classes))
    idxs_classes = [set((np.array(dataset.targets) == cls).nonzero()[0]) for cls in range(num_classes)]
    # 创建一个包含所有类别的集合 classes，以及一个列表 idxs_classes，其中包含每个类别的样本索引集合。
    num_images = len(dataset)
    # 获取数据集中的图像总数。
    while(1):
        # 开始一个无限循环，将在满足条件时通过 break 语句退出。
        for cls in range(num_classes):
            # 遍历所有类别。
            if len(idxs_classes[cls]) > 0:
                # 如果当前类别还有未分配的样本。
                for client_id in range(num_clients):
                    # 遍历所有客户端。
                    if num_images_clients[client_id] > 0:
                        # 如果当前客户端还有图像分配额度。
                        num_images_client_class = min((q_class_tilde[client_id, cls] * num_images_clients[client_id]).round().to(int).item(), len(idxs_classes[cls]))
                        # 计算当前客户端和类别应该分配的图像数量。
                        idxs_client_class = list(np.random.choice(list(idxs_classes[cls]), num_images_client_class, replace=False))
                        # 从当前类别的未分配样本中随机选择 num_images_client_class 个样本，形成客户端的样本索引列表。
                        split_idxs[client_id] += idxs_client_class
                        # 将选中的样本索引添加到当前客户端的索引列表。
                        idxs_classes[cls] -= set(idxs_client_class)
                        # 从当前类别的未分配样本集合中移除已分配的样本。
                        num_images_clients[client_id] -= num_images_client_class
                        # 更新客户端的图像分配额度。
                        split_dist[client_id, cls] += num_images_client_class
                        # 更新拆分分布矩阵。
                        if len(idxs_classes[cls]) == 0 and len(classes) > 1:
                            # 如果当前类别的样本已全部分配完毕，并且还有其他类别未分配完毕。
                            classes -= {cls}
                            # 从类别集合中移除当前类别。
                            q_class_tilde[:, cls] = 0
                            # 将 q_class_tilde 矩阵中当前类别的所有行设置为 0。
                            idxs = (q_class_tilde == 0).all(1)
                            # 找出 q_class_tilde 矩阵中所有元素为 0 的行的索引。
                            q_class_tilde[idxs, list(classes)[0]] = 1
                            # 将这些行为新的主要类别。
                            q_class_tilde /= q_class_tilde.sum(1, keepdim=True)
                            # 重新归一化 q_class_tilde 矩阵。
                            break
                            # 退出当前类别的循环。
        if num_images_clients.sum() == 0: break
        # 如果所有客户端的图像分配额度都已用完。 退出循环。
    split_emd = {}
    # 初始化一个字典 split_emd，用于存储拆分的 EMD 值。
    split_emd['class'] = earthmover_distance(split_dist)
    # 计算类别分布的 EMD 值，并将其存储在 split_emd 字典中。
    split_emd['client'] = torch.abs(split_dist.sum(1)/split_dist.sum() - torch.tensor([1/num_clients]*num_clients)).sum()
    # 计算客户端分布的 EMD 值，并将其存储在 split_emd 字典中。
    return Split(split_idxs, split_dist, split_emd)
    # 创建一个 Split 对象，包含客户端索引、拆分分布和 EMD 值，并返回该对象。


# 这个 get_splits 函数通过指定的 IID 和平衡性参数生成数据集的拆分，
# 使得可以模拟不同的数据分布情况。这对于联邦学习等分布式学习场景非常有用。
def get_splits(datasets, num_clients, iid, balance):
    # 这段代码定义了一个名为 get_splits 的函数，用于根据指定的 IID（独立同分布）和平衡性参数生成数据集的拆分。
    # 以下是对函数及其参数的详细解释：
    # 定义了一个名为 get_splits 的函数，它接收以下参数：
    # datasets：一个字典，包含不同数据集类型（如训练集、验证集、测试集）。
    # num_clients：客户端的数量。
    # iid：表示数据集是否独立同分布（IID）的参数。
    # balance：表示数据集分布的平衡性参数。
    num_classes = len(datasets['train'].classes)
    # 获取训练数据集的类别数量。
    if iid == 0:
        # 如果 iid 参数为 0，表示每个客户端的类别分布是完全随机的。
        q_class = torch.zeros((num_clients, num_classes))
        for client_id in range(num_clients):
            q_class[client_id, np.random.randint(low=0, high=num_classes)] = 1
        # 为每个客户端创建一个零矩阵 q_class，然后在每个客户端随机选择一个类别并将对应位置设置为 1。
    elif iid == float('inf'):
        # 如果 iid 参数为无穷大，表示所有客户端的类别分布与全局分布相同。
        q_class = torch.tensor([(datasets['train'].targets == cls).sum() for cls in range(len(datasets['train'].classes))])
        # 计算每个类别在训练数据集中的样本数量。
        q_class = q_class / len(datasets['train'])
        # 将类别分布归一化。
        q_class = q_class.repeat(num_clients, 1)
        # 复制归一化的类别分布，为每个客户端创建相同的分布。
    else:
        # 如果 iid 参数不是 0 或无穷大，表示类别分布介于完全随机和完全相同之间。
        p_class = torch.tensor([(datasets['train'].targets == cls).sum() for cls in range(len(datasets['train'].classes))])
        # 计算每个类别在训练数据集中的样本数量。
        p_class = p_class / len(datasets['train'])
        # 将类别分布归一化。
        q_class = torch.distributions.dirichlet.Dirichlet(iid*p_class).sample((num_clients,))
        # 使用狄利克雷分布生成每个客户端的类别分布。
    if balance == 0:
        # 如果 balance 参数为 0，表示客户端分布是完全随机的。
        q_client = torch.zeros(num_clients).reshape((num_clients, 1))
        q_client[np.random.randint(low=0, high=num_clients)] = 1
        # 为客户端创建一个零矩阵 q_client，然后在随机位置设置为 1。
    elif balance == float('inf'):
        # 如果 balance 参数为无穷大，表示所有客户端的样本分布是均匀的。
        q_client = (torch.ones(num_clients).divide(num_clients)).reshape((num_clients, 1))
        # 创建一个均匀分布的客户端分布。
    else:
        # 如果 balance 参数不是 0 或无穷大，表示客户端分布介于完全随机和完全均匀之间。
        p_client = torch.ones(num_clients).divide(num_clients)
        # 创建一个均匀分布的客户端分布。
        q_client = torch.distributions.dirichlet.Dirichlet(balance*p_client).sample().reshape((num_clients,1))
        # 使用狄利克雷分布生成客户端分布。
    splits = {}
    # 初始化一个空字典 splits，用于存储不同数据集类型的拆分。
    for key in datasets.keys():
        # 遍历 datasets 字典中的键。
        splits[key] = get_split(datasets[key], q_class, q_client)
        # 对于每个数据集类型，调用 get_split 函数生成拆分，并将其存储在 splits 字典中。
    return splits
    # 返回包含不同数据集类型拆分的 splits 字典。


# 这个 get_splits_fig 函数通过可视化的方式展示了不同数据集类型中每个类别在不同客户端的分布情况，有助于理解数据拆分是否均匀。
# 通过调整 iid 和 balance 参数，可以控制数据集的分布特性。
def get_splits_fig(splits, iid, balance):
    # 这段代码定义了一个名为 get_splits_fig 的函数，用于可视化数据集拆分的类分布。以下是对函数及其参数的详细解释：
    # 定义了一个名为 get_splits_fig 的函数，它接收三个参数：
    # splits：一个字典，包含不同数据集类型（如训练集、验证集、测试集）的分布信息。
    # iid：表示数据集是否独立同分布（Independent and Identically Distributed，IID）的值。
    # balance：表示数据集分布的平衡性。
    types, titles = [], []
    # 初始化两个列表，types 用于存储数据集类型的名称，titles 用于存储数据集类型的标题。
    for type in splits:
        if splits[type] is not None:
            types.append(type)
            titles.append(type.capitalize())
    # 遍历 splits 字典，如果某个类型的数据集不为空，则将其名称添加到 types 列表，并将其首字母大写的名称添加到 titles 列表。
    fig, ax = plt.subplots(1, len(types))
    # 创建一个图形 fig 和一个包含 1 行 len(types) 列的子图轴对象 ax。
    iid_str = '∞' if iid == float('inf') else '%g' % iid
    balance_str = '∞' if balance == float('inf') else '%g' % balance
    # 格式化 iid 和 balance 的显示字符串，如果它们是无穷大（float('inf')），则用 ∞ 表示，否则格式化为一般数字。
    num_clients, num_classes = splits['train'].dist.shape
    # 获取训练集分布矩阵 splits['train'].dist 的形状，num_clients 表示客户端数量，num_classes 表示类别数量。
    y = torch.arange(num_clients)
    # 创建一个从 0 到 num_clients-1 的整数序列 y。
    for i, type in enumerate(types):
        # 遍历 types 列表中的每个数据集类型。
        left = torch.zeros(num_clients)
        # 初始化一个 left 变量，用于累加每个类别的分布。
        for c in range(num_classes):
            ax[i].barh(y, splits[type].dist[:,c], left=left, height=1)
            left += splits[type].dist[:,c]
            # 对于每个类别，使用 barh 函数在子图 ax[i] 中绘制水平条形图，表示该类别在不同客户端的分布。
            # left=left 参数用于累加分布，height=1 表示条形的高度。
        ax[i].set_xlim((0, max(left)))
        # 设置子图 ax[i] 的 x 轴显示范围。
        ax[i].set_xlabel('Class distribution')
        # 设置子图 ax[i] 的 x 轴标签为 "Class distribution"。
        ax[i].set_title(titles[i])
        # 设置子图 ax[i] 的标题为 titles[i]。
        if i == 0:
            ax[i].set_ylabel('Client')
        else:
            ax[i].set_yticks([])
        # 如果是第一个子图，则设置 y 轴标签为 "Client"；否则，不显示 y 轴刻度。
    fig.suptitle('$α_{class} = %s, α_{client} = $%s' % (iid_str, balance_str))
    # 设置整个图形的标题，包括 iid 和 balance 的值。
    fig.tight_layout()
    # 自动调整子图参数，使之填充整个图形区域。
    fig.set_size_inches(4*len(types), 4)
    # 设置图形的大小。
    return fig
    # 返回创建的图形对象。
