#!/usr/bin/env python
# encoding: utf-8
import os

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred = np.squeeze(pred)
    true = np.squeeze(true)
    return mean_absolute_percentage_error(np.array(true) + 1, np.array(pred) + 1)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape


def metric_rolling(name, pre, gt, rm=16, inter=72):
    # rm indicates the number of days to be rolled, inter indicates the interval
    pre = pre.detach().cpu().numpy() if hasattr(pre, 'detach') else pre
    gt = gt.detach().cpu().numpy() if hasattr(gt, 'detach') else gt

    pre, gt = pre.flatten(), gt.flatten()

    # pre = np.array(pre)
    # gt = np.array(gt)
    ll = int(len(pre) / inter)
    pre_all = []
    gt_all = []
    for i in range(ll):
        pre_all.extend(pre[i * inter: (i * inter + rm)])
        gt_all.extend(gt[i * inter: (i * inter + rm)])
    mae, mse, rmse, mape = metric(np.array(pre_all), np.array(gt_all))

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }


    return metrics

