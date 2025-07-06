import random
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale
import config


def get_batch(data, split="train"):
    batch_size = config.BATCH_SIZE
    window_size = config.WINDOW_SIZE
    data_unhealthy_train, data_healthy_train = data[0]
    data_unhealthy_val, data_healthy_val = data[1]
    data_unhealthy_test, data_healthy_test = data[2]
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
    batch_y = []
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
        batch_y.append(0.1)

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


def get_batch_3(data_train, data_val, data_test, split="train"):
    batch_size = config.BATCH_SIZE
    window_size = config.WINDOW_SIZE
    data_unhealthy_train, data_healthy_train = data_train
    data_unhealthy_val, data_healthy_val = data_val
    data_unhealthy_test, data_healthy_test = data_test
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
