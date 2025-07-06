from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import wfdb
import time
import random
from sklearn.preprocessing import minmax_scale
import sys
from torch.utils.tensorboard import SummaryWriter



def intersection(lst1, lst2):

    return list(set(lst1) & set(lst2))


def move_to(patient_id, source, target):
    moved_files = []

    for file_ in source:
        if file_[:10] == patient_id:
            moved_files.append(file_)

    for file_ in moved_files:
        source.remove(file_)
        target.append(file_)


def de_intersection(src1, src2):
    ids1 = [element[:10] for element in src1]
    ids2 = [element[:10] for element in src2]
    intersection_id = intersection(ids1, ids2)
    move_to_src1 = intersection_id[: int(0.5 * len(intersection_id))]
    move_to_src2 = intersection_id[int(0.5 * len(intersection_id)) :]
    for id in move_to_src1:
        move_to(id, src2, src1)
    for id in move_to_src2:
        move_to(id, src1, src2)


# def load_gen_data(result_path):
#     """加载保存的训练/验证/测试数据列表"""
#     # 加载数据列表
#     hc_train = np.load(os.path.join(result_path, 'hc_train.npy'), allow_pickle=True).tolist()
#     hc_val = np.load(os.path.join(result_path, 'hc_val.npy'), allow_pickle=True).tolist()
#     hc_test = np.load(os.path.join(result_path, 'hc_test.npy'), allow_pickle=True).tolist()
#     mi_train = np.load(os.path.join(result_path, 'mi_train.npy'), allow_pickle=True).tolist()
#     mi_val = np.load(os.path.join(result_path, 'mi_val.npy'), allow_pickle=True).tolist()
#     mi_test = np.load(os.path.join(result_path, 'mi_test.npy'), allow_pickle=True).tolist()
    
#     return hc_train, hc_val, hc_test, mi_train, mi_val, mi_test


def load_gen_data2():
    """加载保存的训练/验证/测试数据列表"""
    # 加载数据列表
    result_path = r'result_data\ptbdb\42\self_onn\2025-04-14-01-03-49\data'
    hc_train = np.load(os.path.join(result_path, 'hc_train.npy'), allow_pickle=True).tolist()
    hc_val = np.load(os.path.join(result_path, 'hc_val.npy'), allow_pickle=True).tolist()
    hc_test = np.load(os.path.join(result_path, 'hc_test.npy'), allow_pickle=True).tolist()
    mi_train = np.load(os.path.join(result_path, 'mi_train.npy'), allow_pickle=True).tolist()
    mi_val = np.load(os.path.join(result_path, 'mi_val.npy'), allow_pickle=True).tolist()
    mi_test = np.load(os.path.join(result_path, 'mi_test.npy'), allow_pickle=True).tolist()
    
    return hc_train, hc_val, hc_test, mi_train, mi_val, mi_test

def gen_data(result_path,seed_num, chns=None):

    # load real data (ptbdb)
    with open("ptbdb_data/RECORDS") as fp:
        lines = fp.readlines()

    files_mi, files_hc = [], []

    for file in lines:
        file_path = "ptbdb_data/" + file[:-1] + ".hea"  # 读取头文件

        # read header to determine class
        if "Myocardial infarction" in open(file_path).read():
            files_mi.append(file)

        if "Healthy control" in open(file_path).read():
            files_hc.append(file)

    # shuffle data (cross-validation)
    np.random.seed(int(seed_num))
    np.random.shuffle(files_mi)
    np.random.shuffle(files_hc)
    # 划分train、valid、test 文件名
    # train_rate = config.TRAIN_RATE
    train_rate = 0.8
    hc_train = files_hc[: int(train_rate * len(files_hc))]
    hc_val_test = files_hc[int(train_rate * len(files_hc)) :]
    mi_train = files_mi[: int(train_rate * len(files_mi))]
    mi_val_test = files_mi[int(train_rate * len(files_mi)) :]

    de_intersection(hc_train, hc_val_test)
    de_intersection(mi_train, mi_val_test)

    hc_val = hc_val_test[: int(0.5 * len(hc_val_test))]
    hc_test = hc_val_test[int(0.5 * len(hc_val_test)) :]
    mi_val = mi_val_test[: int(0.5 * len(mi_val_test))]
    mi_test = mi_val_test[int(0.5 * len(mi_val_test)) :]

    de_intersection(hc_val, hc_test)
    de_intersection(mi_val, mi_test)

    chns = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"] if chns == "ALL" else chns
    path = os.path.join(result_path, 'data')
    # hc_train, hc_val, hc_test, mi_train, mi_val, mi_test = load_gen_data(path)
    data_hc_train = []
    data_hc_val = []
    data_hc_test = []
    data_mi_train = []
    data_mi_val = []
    data_mi_test = []
    for file in hc_train:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_hc_train.append(data)

    for file in hc_val:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_hc_val.append(data)

    for file in hc_test:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_hc_test.append(data)

    for file in mi_train:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_mi_train.append(data)

    for file in mi_val:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_mi_val.append(data)

    for file in mi_test:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_mi_test.append(data)
    print(hc_train)
    print(hc_val)
    print(hc_test)
    print(mi_train)
    print(mi_val)    
    print(mi_test)
    # path = os.path.join(result_path, 'data')
  

    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, 'hc_train.npy'), hc_train)
    np.save(os.path.join(path, 'hc_val.npy'), hc_val)
    np.save(os.path.join(path, 'hc_test.npy'), hc_test)
    np.save(os.path.join(path, 'mi_train.npy'), mi_train)
    np.save(os.path.join(path, 'mi_val.npy'), mi_val)
    np.save(os.path.join(path, 'mi_test.npy'), mi_test)

    

    data_hc_train = np.array(data_hc_train, dtype=object)
    data_hc_val = np.array(data_hc_val, dtype=object)
    data_hc_test = np.array(data_hc_test, dtype=object)
    data_mi_train = np.array(data_mi_train, dtype=object)
    data_mi_val = np.array(data_mi_val, dtype=object)
    data_mi_test = np.array(data_mi_test, dtype=object)
    data_train = (data_hc_train, data_mi_train)
    data_val = (data_hc_val, data_mi_val)
    data_test = (data_hc_test, data_mi_test)
    return [data_train, data_val, data_test]




def get_batch(batch_size,train_data,val_data,test_data,window_size=10000,split='train'):
    # batch_size = config.BATCH_SIZE
    # window_size = config.WINDOW_SIZE
    batch_size = batch_size
    data_unhealthy_train, data_healthy_train = train_data
    data_unhealthy_val, data_healthy_val = val_data
    data_unhealthy_test, data_healthy_test = test_data
    if split == "train":
        unhealthy_indices = random.sample(list(np.arange(len(data_unhealthy_train))), k=int(batch_size / 2))
        healthy_indices = random.sample(list(np.arange(len(data_healthy_train))), k=int(batch_size / 2))
        mi_batch = data_unhealthy_train[unhealthy_indices]
        hc_batch = data_healthy_train[healthy_indices]
    elif split == "val":
        unhealthy_indices = random.sample(list(np.arange(len(data_unhealthy_val))), k=int(batch_size / 2))
        healthy_indices = random.sample(list(np.arange(len(data_healthy_val))), k=int(batch_size / 2))
        mi_batch = data_unhealthy_val[unhealthy_indices]
        hc_batch = data_healthy_val[healthy_indices]
    elif split == "test":
        unhealthy_indices = random.sample(list(np.arange(len(data_unhealthy_test))), k=int(batch_size / 2))
        healthy_indices = random.sample(list(np.arange(len(data_healthy_test))), k=int(batch_size / 2))
        mi_batch = data_unhealthy_test[unhealthy_indices]
        hc_batch = data_healthy_test[healthy_indices]

    batch_x = []
    chn_num = mi_batch.shape[1]
    for sample in mi_batch:

        start = random.choice(np.arange(len(sample[0]) - window_size))

        # normalize
        # normalized_1 = minmax_scale(sample[0][start : start + window_size])
        # normalized_2 = minmax_scale(sample[1][start : start + window_size])
        # normalized = np.array((normalized_1, normalized_2))

        normalized_list = []
        for i in range(chn_num):
            normalized_list.append(minmax_scale(sample[i][start : start + window_size]))
        normalized = np.array(normalized_list)

        batch_x.append(normalized)

    for sample in hc_batch:

        start = random.choice(np.arange(len(sample[0]) - window_size))

        # normalize
        # normalized_1 = minmax_scale(sample[0][start : start + window_size])
        # normalized_2 = minmax_scale(sample[1][start : start + window_size])
        # normalized = np.array((normalized_1, normalized_2))
        normalized_list = []
        for i in range(chn_num):
            normalized_list.append(minmax_scale(sample[i][start : start + window_size]))
        normalized = np.array(normalized_list)

        batch_x.append(normalized)

    # 0.1 for unhealthy, 0.9 for healthy
    batch_y = [0.1 for _ in range(int(batch_size / 2))]
    for _ in range(int(batch_size / 2)):
        batch_y.append(0.9)

    indices = np.arange(len(batch_y))
    np.random.shuffle(indices)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    batch_x = batch_x[indices]
    batch_y = batch_y[indices]

    batch_x = np.reshape(batch_x, (-1, chn_num, window_size))
    batch_x = torch.from_numpy(batch_x)
    batch_x = batch_x.float().cuda()
    batch_x = batch_x.float()

    batch_y = np.reshape(batch_y, (-1, 1))
    batch_y = torch.from_numpy(batch_y)
    batch_y = batch_y.float().cuda()
    batch_y = batch_y.float()

    return batch_x, batch_y





def get_batch_0409(batch_size,data_unhealthy_train,data_healthy_train,data_unhealthy_val,data_healthy_val,data_unhealthy_test,data_healthy_test,window_size=10000,split='train'):
    # batch_size = config.BATCH_SIZE
    # window_size = config.WINDOW_SIZE
    batch_size = batch_size
    # data_unhealthy_train, data_healthy_train 
    # data_unhealthy_val, data_healthy_val
    # data_unhealthy_test, data_healthy_test 
    if split == "train":
        unhealthy_indices = random.sample(list(np.arange(len(data_unhealthy_train))), k=int(batch_size / 2))
        healthy_indices = random.sample(list(np.arange(len(data_healthy_train))), k=int(batch_size / 2))
        mi_batch = data_unhealthy_train[unhealthy_indices]
        hc_batch = data_healthy_train[healthy_indices]
    elif split == "val":
        unhealthy_indices = random.sample(list(np.arange(len(data_unhealthy_val))), k=int(batch_size / 2))
        healthy_indices = random.sample(list(np.arange(len(data_healthy_val))), k=int(batch_size / 2))
        mi_batch = data_unhealthy_val[unhealthy_indices]
        hc_batch = data_healthy_val[healthy_indices]
    elif split == "test":
        unhealthy_indices = random.sample(list(np.arange(len(data_unhealthy_test))), k=int(batch_size / 2))
        healthy_indices = random.sample(list(np.arange(len(data_healthy_test))), k=int(batch_size / 2))
        mi_batch = data_unhealthy_test[unhealthy_indices]
        hc_batch = data_healthy_test[healthy_indices]

    batch_x = []
    chn_num = mi_batch.shape[1]
    for sample in mi_batch:

        start = random.choice(np.arange(len(sample[0]) - window_size))

        # normalize
        # normalized_1 = minmax_scale(sample[0][start : start + window_size])
        # normalized_2 = minmax_scale(sample[1][start : start + window_size])
        # normalized = np.array((normalized_1, normalized_2))

        normalized_list = []
        for i in range(chn_num):
            normalized_list.append(minmax_scale(sample[i][start : start + window_size]))
        normalized = np.array(normalized_list)

        batch_x.append(normalized)

    for sample in hc_batch:

        start = random.choice(np.arange(len(sample[0]) - window_size))

        # normalize
        # normalized_1 = minmax_scale(sample[0][start : start + window_size])
        # normalized_2 = minmax_scale(sample[1][start : start + window_size])
        # normalized = np.array((normalized_1, normalized_2))
        normalized_list = []
        for i in range(chn_num):
            normalized_list.append(minmax_scale(sample[i][start : start + window_size]))
        normalized = np.array(normalized_list)

        batch_x.append(normalized)

    # 0.1 for unhealthy, 0.9 for healthy
    batch_y = [0.1 for _ in range(int(batch_size / 2))]
    for _ in range(int(batch_size / 2)):
        batch_y.append(0.9)

    indices = np.arange(len(batch_y))
    np.random.shuffle(indices)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    batch_x = batch_x[indices]
    batch_y = batch_y[indices]

    batch_x = np.reshape(batch_x, (-1, chn_num, window_size))
    batch_x = torch.from_numpy(batch_x)
    batch_x = batch_x.float().cuda()
    batch_x = batch_x.float()

    batch_y = np.reshape(batch_y, (-1, 1))
    batch_y = torch.from_numpy(batch_y)
    batch_y = batch_y.float().cuda()
    batch_y = batch_y.float()

    return batch_x, batch_y




def gen_data2(chns=None):

    chns = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"] if chns == "ALL" else chns
    # path = os.path.join(result_path, 'data')
    hc_train, hc_val, hc_test, mi_train, mi_val, mi_test = load_gen_data2()
    data_hc_train = []
    data_hc_val = []
    data_hc_test = []
    data_mi_train = []
    data_mi_val = []
    data_mi_test = []
    for file in hc_train:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_hc_train.append(data)

    for file in hc_val:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_hc_val.append(data)

    for file in hc_test:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_hc_test.append(data)

    for file in mi_train:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_mi_train.append(data)

    for file in mi_val:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_mi_val.append(data)

    for file in mi_test:
        data = []
        for chn in chns:
            data.append(wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(chn)])[0].flatten())
        data_mi_test.append(data)


    data_hc_train = np.array(data_hc_train, dtype=object)
    data_hc_val = np.array(data_hc_val, dtype=object)
    data_hc_test = np.array(data_hc_test, dtype=object)
    data_mi_train = np.array(data_mi_train, dtype=object)
    data_mi_val = np.array(data_mi_val, dtype=object)
    data_mi_test = np.array(data_mi_test, dtype=object)
    data_train = (data_hc_train, data_mi_train)
    data_val = (data_hc_val, data_mi_val)
    data_test = (data_hc_test, data_mi_test)
    return [data_train, data_val, data_test]