#!/usr/bin/env python
# encoding: utf-8
import os
import numpy as np
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import math

import matplotlib.pyplot as plt
from scipy.stats import norm, kruskal
from scipy.stats import skew, kurtosis

from torch.utils.data import TensorDataset

def initial_seed(seed: int = 10):
    """ Fix seed for random number generator. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def parse_kv_argfile(file_path):
    """将 key|value 格式的参数文件转换为 argparse 可识别的参数列表"""
    args_list = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("{") or line.startswith("}"):
            continue
        if "|" in line:
            key, value = line.split("|", 1)
            key = key.strip()
            value = value.strip()
            args_list.append(f"--{key}")
            args_list.append(value)
    return args_list


def r_log_std_normalization(sensor_data_val):
    data = sensor_data_val

    data1 = data[1:]
    data2 = [0 for _ in data1]
    for i in range(len(data) - 1):
        if data[i] > 0:
            data2[i] = data1[i] - data[i]
        else:

            data2[i] = (data1[i] + 1e-8) - (data[i] + 1e-8)
    data = data2

    c = np.array([1] + data)
    mean = np.nanmean(c)
    print("mean is: ", mean)
    std  = np.nanstd(c)
    print("std is ", std)
    c = (c - mean) / std

    mini = 0
    return c, mean, std, mini


def normalize_diff_with_stats(sensor_data_val, mean, std):
    """ Normalize the sensor data using a first-order difference and z-score normalization."""

    data = sensor_data_val
    # diff
    data1 = data[1:]
    data2 = [0 for i in data1]
    for i in range(len(data) - 1):
        if data[i] > 0:
            data2[i] = data1[i] - data[i]
        else:
            data2[i] = (data1[i] + 1e-8) - (data[i] + 1e-8)
    data = data2

    c = np.array([1] + data)

    # norm
    c = (c - mean) / std
    return c


def r_log_std_denorm_dataset(mean, std, predict_y0, y_pre):

    # de-norm
    a2 = predict_y0
    a2 = [ii * std + mean for ii in a2]
    a3 = np.zeros(len(a2))
    a3[0] = a2[0] + y_pre
    for ii in range(len(a2) - 1):
        a3[ii + 1] = a3[ii] + a2[ii + 1]
    return a3


def std_denorm_dataset(predict_y0, pre_y, mean, std):

    a2 = r_log_std_denorm_dataset(mean, std, 0, predict_y0, pre_y)

    return a2


def log_std_normalization(sensor_data_val):
    a = np.log(np.array(sensor_data_val) + 1)
    c = a
    mean = np.nanmean(c)
    # print("mean is: ", mean)
    std = np.nanstd(c)
    # print("std is ", std)
    c = (c - mean) / std

    return c, mean, std

def log_std_normalization_with_stats(sensor_data_val, mean=None, std=None):


    c = np.log(np.array(sensor_data_val) + 1)
    c = (c - mean) / std

    return c


def standard_normalization(x):

    x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    x_norm = (x - mean) / std
    return x_norm, mean, std


def standard_denormalization(x_norm, mean, std):

    x = x_norm * std + mean
    return x


def standard_normalization_with_stats(sensor_data, mean=None, std=None):

    if mean is None or std is None:
        mean = np.mean(sensor_data)
        std = np.std(sensor_data)

    sensor_data_norm = (sensor_data - mean) / std

    if mean is None or std is None:
        return sensor_data_norm, mean, std
    else:
        return sensor_data_norm



def diff_order_1(data):
    a = data
    b = a[:-1]
    a = a[1:] - b
    c = np.array([0] + a.tolist())
    #     d = np.nanmin(c)
    #     c = c - d

    return c

def gen_month_tag(sensor_data):

    sensor_month = sensor_data["datetime"].str[5:7]

    a = sensor_month.str[:]

    a = a.astype(int)

    tag = np.array(a.fillna(np.nan))

    tag = -1 * tag

    return tag


# generate time feature as month+day+hour, transfer str to int, then we have a sequence of meaningful int
def gen_time_feature(sensor_data):

    sensor_month = sensor_data["datetime"].str[5:7]
    sensor_day = sensor_data["datetime"].str[8:10]
    sensor_hour = sensor_data["datetime"].str[11:13]

    #     a = sensor_month.str[:] + sensor_day.str[:] + sensor_hour.str[:]
    #     a = a.astype(int)
    #     b = np.array(a.fillna(np.nan))
    month = sensor_month.astype(np.int8)
    month = np.array(month.fillna(np.nan))
    day = sensor_day.astype(np.int8)
    day = np.array(day.fillna(np.nan))
    hour = sensor_hour.astype(np.int8)
    hour = np.array(hour.fillna(np.nan))

    return month, day, hour


def cos_date(month, day, hour):

    t = []

    for i in range(len(month)):

        #         a = math.cos(((month[i] - 1) * 30.5 * 24 + day[i]*24 + hour[i]) * 2 * (math.pi) / (365 * 24))
        a = math.cos(((month[i] - 1) * 30.5 + day[i]) * 2 * (math.pi) / 365)
        t.append(a)

    return t


def sin_date(month, day, hour):

    t = []

    for i in range(len(month)):

        #         a = math.sin(((month[i] - 1) * 30.5 * 24 + day[i] * 24) * 2 * (math.pi) / (365 * 24))
        a = math.sin(((month[i] - 1) * 30.5 + day[i]) * 2 * (math.pi) / 365)
        t.append(a)

    return t


class RnnDataset(TensorDataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)