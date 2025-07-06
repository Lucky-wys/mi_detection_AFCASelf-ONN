
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



def get_data(seed_num, channel_1, channel_2):
    with open('ptbdb_data/RECORDS') as fp:  
        lines = fp.readlines()

    files_unhealthy, files_healthy = [], []

    for file in lines:
        file_path = "ptbdb_data/" + file[:-1] + ".hea"     # 读取头文件
        
        # read header to determine class
        if 'Myocardial infarction' in open(file_path).read():
            files_unhealthy.append(file)
            
        if 'Healthy control' in open(file_path).read():
            files_healthy.append(file)

    # shuffle data (cross-validation)
    np.random.seed(int(seed_num))
    np.random.shuffle(files_unhealthy)
    np.random.shuffle(files_healthy)

    # 划分train、valid 文件名
    healthy_train = files_healthy[:int(0.8*len(files_healthy))]
    healthy_val = files_healthy[int(0.8*len(files_healthy)):]
    unhealthy_train = files_unhealthy[:int(0.8*len(files_unhealthy))]
    unhealthy_val = files_unhealthy[int(0.8*len(files_unhealthy)):]


    def intersection(lst1, lst2): 
        return list(set(lst1) & set(lst2)) 

    patient_ids_unhealthy_train = [element[:10] for element in unhealthy_train]
    patient_ids_unhealthy_val = [element[:10] for element in unhealthy_val]
    patient_ids_healthy_train = [element[:10] for element in healthy_train]
    patient_ids_healthy_val = [element[:10] for element in healthy_val]

    intersection_unhealthy = intersection(patient_ids_unhealthy_train, patient_ids_unhealthy_val)
    intersection_healthy = intersection(patient_ids_healthy_train, patient_ids_healthy_val)



    # unhealthy
    move_to_train = intersection_unhealthy[:int(0.5*len(intersection_unhealthy))]
    move_to_val = intersection_unhealthy[int(0.5*len(intersection_unhealthy)):]

    for patient_id in move_to_train:
        
        in_val = []
        
        # find and remove all files in val
        for file_ in unhealthy_val:
            if file_[:10] == patient_id:
                in_val.append(file_)
                unhealthy_val.remove(file_)
                
        # add to train
        for file_ in in_val:
            unhealthy_train.append(file_)
        
        
    for patient_id in move_to_val:
        
        in_train = []
        
        # find and remove all files in val
        for file_ in unhealthy_train:
            if file_[:10] == patient_id:
                in_train.append(file_)
                unhealthy_train.remove(file_)
                
        # add to train
        for file_ in in_train:
            unhealthy_val.append(file_)
        
        
    # healthy
    move_to_train = intersection_healthy[:int(0.5*len(intersection_healthy))]
    move_to_val = intersection_healthy[int(0.5*len(intersection_healthy)):]

    for patient_id in move_to_train:
        
        in_val = []
        
        # find and remove all files in val
        for file_ in healthy_val:
            if file_[:10] == patient_id:
                in_val.append(file_)
                healthy_val.remove(file_)
                
        # add to train
        for file_ in in_val:
            healthy_train.append(file_)
            

    for patient_id in move_to_val:
        
        in_train = []
        
        # find and remove all files in val
        for file_ in healthy_train:
            if file_[:10] == patient_id:
                in_train.append(file_)
                healthy_train.remove(file_)
                
        # add to train
        for file_ in in_train:
            healthy_val.append(file_)


    data_healthy_train = []
    for file in healthy_train:
        data_v4, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_1)])
        data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
        data = [data_v4.flatten(), data_v5.flatten()]
        data_healthy_train.append(data)
    data_healthy_val = []
    for file in healthy_val:
        data_v4, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_1)])
        data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
        data = [data_v4.flatten(), data_v5.flatten()]
        data_healthy_val.append(data)
    data_unhealthy_train = []
    for file in unhealthy_train:
        data_v4, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_1)])
        data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
        data = [data_v4.flatten(), data_v5.flatten()]
        data_unhealthy_train.append(data)
    data_unhealthy_val = []
    for file in unhealthy_val:
        data_v4, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_1)])
        data_v5, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_2)])
        data = [data_v4.flatten(), data_v5.flatten()]
        data_unhealthy_val.append(data)

    # print(healthy_train)
    # print(healthy_val)
    # print(unhealthy_train)
    # print(unhealthy_val)


    data_healthy_train = np.array(data_healthy_train,dtype=object)
    data_healthy_val = np.array(data_healthy_val,dtype=object)
    data_unhealthy_train = np.array(data_unhealthy_train,dtype=object)
    data_unhealthy_val = np.array(data_unhealthy_val,dtype=object)

    return data_healthy_train,data_healthy_val, data_unhealthy_train,data_unhealthy_val
   


# def normalize_to_minus_one_one(data):
#     """
#     将数据归一化到 (-1, 1) 范围。
#     """
#     min_val = np.min(data)
#     max_val = np.max(data)
#     if max_val == min_val:  # 防止除零
#         return np.zeros_like(data)
#     return 2 * ((data - min_val) / (max_val - min_val)) - 1




def get_batch(batch_size, data_healthy_train,data_healthy_val, data_unhealthy_train,data_unhealthy_val, split='train'):

    window_size = 10000
    # unhealthy_threshold = int(0.8*num_unhealthy)
    # healthy_threshold = int(0.8*num_healthy)
    
    # unhealthy_test_threshold = int(0.9*num_unhealthy)
    # healthy_test_threshold = int(0.9*num_healthy)
    
    if split == 'train':
        unhealthy_indices = random.sample(list(np.arange(len(data_unhealthy_train))), k=int(batch_size / 2))
        healthy_indices = random.sample(list(np.arange(len(data_healthy_train))), k=int(batch_size / 2))
        unhealthy_batch = data_unhealthy_train[unhealthy_indices]
        healthy_batch = data_healthy_train[healthy_indices]
    elif split == 'val': 
        unhealthy_indices = random.sample(list(np.arange(len(data_unhealthy_val))), k=int(batch_size / 2))
        healthy_indices = random.sample(list(np.arange(len(data_healthy_val))), k=int(batch_size / 2))
        unhealthy_batch = data_unhealthy_val[unhealthy_indices]
        healthy_batch = data_healthy_val[healthy_indices]
    
    batch_x = []
    for sample in unhealthy_batch:
        
        start = random.choice(np.arange(len(sample[0]) - window_size))
        
        # normalize
        normalized_1 = minmax_scale(sample[0][start:start+window_size])
        normalized_2 = minmax_scale(sample[1][start:start+window_size])
        # normalized_1 = minmax_scale(sample[0][start:start+window_size],feature_range=(-1,1))
        # normalized_2 = minmax_scale(sample[1][start:start+window_size],feature_range=(-1,1))
        normalized = np.array((normalized_1, normalized_2))
        
        batch_x.append(normalized)
        
    for sample in healthy_batch:
        
        start = random.choice(np.arange(len(sample[0]) - window_size))
        
        # normalize
        normalized_1 = minmax_scale(sample[0][start:start+window_size])
        normalized_2 = minmax_scale(sample[1][start:start+window_size])
        # normalized_1 = minmax_scale(sample[0][start:start+window_size],feature_range=(-1,1))
        # normalized_2 = minmax_scale(sample[1][start:start+window_size],feature_range=(-1,1))
        normalized = np.array((normalized_1, normalized_2))
        
        batch_x.append(normalized)
    
    batch_y = [0.1 for _ in range(int(batch_size / 2))]
    for _ in range(int(batch_size / 2)):
        batch_y.append(0.9)
        
    indices = np.arange(len(batch_y))
    np.random.shuffle(indices)
    
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    
    batch_x = batch_x[indices]
    batch_y = batch_y[indices]
    
    batch_x = np.reshape(batch_x, (-1, 2, window_size))
    batch_x = torch.from_numpy(batch_x)
    #batch_x = batch_x.float().cuda()
    batch_x = batch_x.float()
    
    batch_y = np.reshape(batch_y, (-1, 1))
    batch_y = torch.from_numpy(batch_y)
    #batch_y = batch_y.float().cuda()
    batch_y = batch_y.float()
    
    return batch_x, batch_y
