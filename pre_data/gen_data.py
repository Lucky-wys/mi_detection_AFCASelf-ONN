from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import wfdb
from sklearn.model_selection import train_test_split
import config


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


def gen_data(seed_num, chns=None):

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
    train_rate = config.TRAIN_RATE
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
