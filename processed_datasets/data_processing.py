#!/usr/bin/env python
# encoding: utf-8
import argparse
import os
import sys
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from datetime import datetime, timedelta
from tqdm import tqdm
from util.data import (initial_seed, parse_kv_argfile, r_log_std_normalization, diff_order_1,
                       gen_month_tag, gen_time_feature, cos_date, sin_date, RnnDataset, normalize_diff_with_stats,
                       standard_normalization, standard_denormalization, standard_normalization_with_stats,
                       log_std_normalization, log_std_normalization_with_stats)


class DataGenerate:

    def __init__(self, data_path, args):

        # All data: Train + Validation + Test
        self.all_input_data = pd.read_csv(args.data_path + args.reservoir_sensor + ".tsv", sep="\t")
        self.all_input_data.columns = ["datetime", "value"]
        self.sensor_all_data, self.all_data, self.all_data_time = None, None, None
        self.sensor_all_data_norm, self.sensor_all_data_norm_list = None, None

        self.all_month, self.all_day, self.all_hour = None, None, None
        self.all_tag, self.all_cos_d, self.all_sin_d = None, None, None

        # some parameters
        self.data_lens = args.input_len + args.output_len + 1

        # GMM3 # dataset-wise gmm gm3 for dim=1
        self.gm3 = GaussianMixture(n_components=3)
        self.gm3_train_recover_prob = None
        self.gm3_min_thres, self.gm3_max_thres = None, None

        self.gmm0 = GaussianMixture(n_components=3, )
        self.gmm0_train_recover_prob, self.gmm0_means = None, None

        # sample-wise gmm, generate dim 5-7, using the extreme score(dim=1)
        self.sample_wise_gmm = GaussianMixture(n_components=3, )

        # DANet GMM
        self.dangmm3 = GaussianMixture(n_components=3)

        # Train data
        self.sensor_train_data_norm_list, self.sensor_train_data_norm = None, None
        self.train_data, self.train_data_time = None, None
        self.train_diff_mean, self.train_diff_std, self.train_diff_mini = None, None, None
        self.train_tag, self.train_cos_d, self.train_sin_d = None, None, None
        self.train_month, self.train_day, self.train_hour = None, None, None
        self.Train_DataSets = None
        self.x_train, self.y_train = None, None
        self.train_data_loader = None

        self.x_normal_train, self.y_normal_train = None, None

        self.traditional_train_data_norm, self.stdn_mean, self.stdn_std = None, None, None

        self.traditional_train_data_norm_list, self.ori_trian_data_norm_list = None, None
        self.log_std_train_data_norm_list = None

        self.log_std_train_data_norm, self.logstd_mean, self.logstd_std = None, None, None

        self.dan_gm3_prob_like_outlier3 = None
        self.R_sensor_data_norm, self.R_mean, self.R_std = None, None, None
        self.R_data, self.R_sensor_data_norm1 = None, None

        # Validation data
        self.val_points = []

        # Test data
        self.test_data_index = []

        # Get data
        self.read_train_dataset(args)
        self.get_val_data_index(args)
        self.get_train_data(args)
        self.get_test_data_index(args)

        # self.val_datasets = self.get_batch_data(self.val_points, args)
        # self.test_datasets = self.get_batch_data(self.test_data_index, args)

        val_x, val_y = self.get_batch_data(self.val_points, args)
        test_x, test_y = self.get_batch_data(self.test_data_index, args)


        save_dir = os.path.join(args.outf, args.name, f"in{args.input_len}_out{args.output_len}_ro{args.roll}")
        os.makedirs(save_dir, exist_ok=True)


        np.save(os.path.join(save_dir, "train_x.npy"), self.x_train)
        np.save(os.path.join(save_dir, "train_y.npy"), self.y_train)
        np.save(os.path.join(save_dir, "train_x_normal.npy"), self.x_normal_train)
        np.save(os.path.join(save_dir, "train_y_normal.npy"), self.y_normal_train)
        np.save(os.path.join(save_dir, "val_x.npy"), val_x)
        np.save(os.path.join(save_dir, "val_y.npy"), val_y)
        np.save(os.path.join(save_dir, "test_x.npy"), test_x)
        np.save(os.path.join(save_dir, "test_y.npy"), test_y)

        mean_std_mini = {'diff_mean': self.train_diff_mean, 'diff_std': self.train_diff_std,
                         'mini': self.train_diff_mini, 'stdn_mean': self.stdn_mean, 'stdn_std': self.stdn_std,
                         'logstd_mean': self.logstd_mean, 'logstd_std': self.logstd_std,
                         'R_mean':self.R_mean, 'R_std':self.R_std,}
        torch.save(mean_std_mini, os.path.join(save_dir, "mean_std_mini.pt"))

        print('DataGenerate initialized with reservoir sensor:', args.reservoir_sensor)

    def read_train_dataset(self, args):

        trainX = self.all_input_data[["datetime", "value"]]

        # read sensor data to vector
        train_start_num = trainX[trainX["datetime"] == args.train_start_point].index.values[
            0]  # start_point: start time of the train set
        print("for sensor ", args.reservoir_sensor, "train_start_num is: ", train_start_num)
        # foot label of train_end
        train_length = (trainX[trainX["datetime"] == args.train_end_point].index.values[
                            0] - train_start_num)  # train_point: end time of the train set
        print("train set length is : ", train_length)

        sensor_data = trainX[
                      train_start_num: train_length + train_start_num]
        data = np.array(sensor_data["value"].fillna(np.nan))  # fillna(np.nan) means fill the None value with np.nan
        diff_data = diff_order_1(data)  # diff_order_1: calculate the first order difference of the data, i.e. data[i] - data[i-1]
        data_time = np.array(sensor_data["datetime"].fillna(np.nan))  # fill the None value with np.nan
        sensor_train_data_norm, diff_mean, diff_std, diff_mini = r_log_std_normalization(data)  #
        traditional_train_data_norm, stdn_mean, stdn_std = standard_normalization(data)
        log_std_train_data_norm, logstd_mean, logstd_std = log_std_normalization(data)

        sensor_train_data_norm_list = [[ff] for ff in sensor_train_data_norm]
        traditional_train_data_norm_list = [[ff] for ff in traditional_train_data_norm]
        log_std_train_data_norm_list = [[ff] for ff in log_std_train_data_norm]

        ori_trian_data_norm_list = [[ff] for ff in data]

        # save the time and diff_log_std_norm data
        self.train_data, self.train_data_time = data, data_time
        self.sensor_train_data_norm = sensor_train_data_norm

        self.train_diff_mean, self.train_diff_std, self.train_diff_mini = diff_mean, diff_std, diff_mini
        self.traditional_train_data_norm, self.stdn_mean, self.stdn_std = traditional_train_data_norm, stdn_mean, stdn_std
        self.log_std_train_data_norm, self.logstd_mean, self.logstd_std = log_std_train_data_norm, logstd_mean, logstd_std

        self.traditional_train_data_norm_list, self.ori_trian_data_norm_list = traditional_train_data_norm_list, ori_trian_data_norm_list
        self.log_std_train_data_norm_list = log_std_train_data_norm_list

        # MCANN gmm
        gmm_input = sensor_train_data_norm

        clean_data = []
        for ii in range(len(sensor_train_data_norm)):
            if (sensor_train_data_norm[ii] is not None) and (np.isnan(sensor_train_data_norm[ii]) != 1):
                clean_data.append(sensor_train_data_norm[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)  # 将一维数组转化为二维数组 (N, 1)

        # dataset-wise gmm gm3 for dim=1
        self.gm3.fit(sensor_data_prob)

        save_dir = os.path.join(args.outf, args.name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.gm3, os.path.join(save_dir, "train_GM3.pt"))

        gm_means = np.squeeze(self.gm3.means_)
        gm3_z0 = np.min(gm_means)
        gm3_z1 = np.median(gm_means)
        gm3_z2 = np.max(gm_means)

        gm3_thre1 = (gm3_z0 + gm3_z1) / 2
        gm3_thre2 = (gm3_z1 + gm3_z2) / 2

        self.gm3_min_thres, self.gm3_max_thres = gm3_thre1, gm3_thre2

        print("gm3.means are: ", gm_means)
        print("gm3 thresholds are: {} {}, and min, median, max are: {} {}, {}".format(gm3_thre1, gm3_thre2, gm3_z0, gm3_z1, gm3_z2))  # 打印阈值
        print("gm3.covariances are: {}, and gm3.weights are: {}".format(self.gm3.covariances_, self.gm3.weights_))

        gm3_weights = self.gm3.weights_
        gm3_prob3 = self.gm3.predict_proba(sensor_data_prob)

        # computing a score to gently highlight the extreme values
        gm3_prob_in_distribution = (gm3_prob3[:, 0] * gm3_weights[0] +
                                    gm3_prob3[:, 1] * gm3_weights[1] +
                                    gm3_prob3[:, 2] * gm3_weights[2])
        gm3_prob_like_outlier = 1 - gm3_prob_in_distribution
        gm3_prob_like_outlier = gm3_prob_like_outlier.reshape((len(sensor_data_prob), 1))  # shape(训练集长度, 1)

        recover_data = []
        temp = 0
        jj = 0
        for ii in range(len(sensor_train_data_norm)):
            if (sensor_train_data_norm[ii] is not None) and (np.isnan(sensor_train_data_norm[ii]) != 1):
                recover_data.append(gm3_prob_like_outlier[jj])
                jj = jj + 1
            else:
                recover_data.append(sensor_train_data_norm[ii])
        gm3_prob_like_outlier = np.array(recover_data, np.float32).reshape(len(sensor_train_data_norm), 1)

        self.gm3_train_recover_prob = gm3_prob_like_outlier  # 保存 gm3 的恢复概率


        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, gm3_prob_like_outlier),1)  # dim=1, from gm3  # 把“离群分数”拼成新的特征：in dim=1. (1阶差分归一化，离群分数，)

        clean_data = []
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                clean_data.append(gmm_input[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)

        series = []
        random.seed(args.val_seed)
        for ggg in range(200000):
            g0 = random.randint(0, len(gmm_input) - args.output_len)
            if not np.isnan(gmm_input[g0]).any():
                series.append([gmm_input[g0]])

        self.gmm0.fit(np.array(series).reshape(-1, 1))

        torch.save(self.gmm0, os.path.join(save_dir, "train_GMM0.pt"))

        gmm0_means = np.squeeze(self.gmm0.means_)
        print("gmm0.means are: {}, and gmm0.weights are: {}".format(gmm0_means, self.gmm0.weights_))
        gmm0_weights3 = self.gmm0.weights_


        data_prob30 = self.gmm0.predict_proba(sensor_data_prob)

        order1 = np.argmax(gmm0_weights3)
        d0 = data_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmin(gmm0_weights3)
        d1 = data_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("new order is, ", order1, order2, order3)

        data_prob3 = np.concatenate((d0, d1), 1)
        data_prob3 = np.concatenate((data_prob3, data_prob30[:, order3].reshape(-1, 1)), 1)  # # data_prob3.shape (训练集长度, 3)，每行是一个样本在 3 个成分上的后验概率

        recover_prob = []
        temp = np.zeros(np.array(data_prob3[0]).shape)
        jj = 0
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                recover_prob.append(data_prob3[jj])
                jj = jj + 1
            else:
                recover_prob.append(temp)
        recover_prob = np.array(recover_prob, np.float32)

        self.gmm0_train_recover_prob = recover_prob

        # now, dim 2,3,4 are the ordered posterior
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, recover_prob[:, 0:1]), 1)
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, recover_prob[:, 1:2]), 1)
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, recover_prob[:, 2:3]), 1)

        # dim 5, 6 are the traditional standardized data and original data
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, self.traditional_train_data_norm_list), 1)
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, self.ori_trian_data_norm_list), 1)

        # dim 7, 8, 9, 10, 11 are time features
        data_time_str = data_time.astype(str)
        data_time_pd = pd.to_datetime(data_time_str)

        year = data_time_pd.year
        month = data_time_pd.month
        day = data_time_pd.day
        hour = data_time_pd.hour
        minute = data_time_pd.minute

        time_features = np.stack([year, month, day, hour, minute], axis=1)

        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, time_features), 1)

        # dim 12, 13 DANet GMM and log_std normalization

        dan_clean_data = []
        for ii in range(len(data)):
            if (data[ii] is not None) and (np.isnan(data[ii]) != 1):
                dan_clean_data.append(data[ii])
        dan_sensor_data_prob = np.array(dan_clean_data, np.float32).reshape(-1, 1)

        self.dangmm3.fit(dan_sensor_data_prob)

        save_dir = os.path.join(args.outf, args.name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.dangmm3, os.path.join(save_dir, "train_DAN_GM3.pt"))

        dan_weights3 = self.dangmm3.weights_

        dan_data_prob3 = self.dangmm3.predict_proba(dan_sensor_data_prob)

        dan_gm3_prob_in_distribution3 = (
                dan_data_prob3[:, 0] * dan_weights3[0]
                + dan_data_prob3[:, 1] * dan_weights3[1]
                + dan_data_prob3[:, 2] * dan_weights3[2]
        )

        dan_gm3_prob_like_outlier3 = 1 - dan_gm3_prob_in_distribution3
        dan_gm3_prob_like_outlier3 = dan_gm3_prob_like_outlier3.reshape((len(dan_sensor_data_prob), 1))

        dan_recover_data = []
        temp = 0
        jj = 0
        for ii in range(len(data)):
            if (data[ii] is not None) and (np.isnan(data[ii]) != 1):
                dan_recover_data.append(dan_gm3_prob_like_outlier3[jj])  #
                jj = jj + 1
            else:
                dan_recover_data.append(data[ii])  # None / NaN
        dan_gm3_prob_like_outlier3 = np.array(dan_recover_data, np.float32).reshape(len(data), 1)

        self.dan_gm3_prob_like_outlier3 = dan_gm3_prob_like_outlier3

        # sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, dan_gm3_prob_like_outlier3),1)  # dim=1, from gm3
        sensor_train_data_norm_list = np.concatenate((sensor_train_data_norm_list, self.log_std_train_data_norm_list), 1)  # dim=8, from dan_gmm3

        self.sensor_train_data_norm_list = sensor_train_data_norm_list

        self.R_data = dan_gm3_prob_like_outlier3
        self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(self.R_data)

        # self.R_sensor_data_norm1 = dan_gm3_prob_like_outlier3.squeeze()       # original, can not work
        self.R_sensor_data_norm1 = self.R_sensor_data_norm.squeeze()
        self.R_sensor_data_norm = self.R_sensor_data_norm1  # shape (len(self.data),)

        print("sensor_train_data_norm_list, ", sensor_train_data_norm_list)
        print("Finish prob indicator generating.")

        tag = gen_month_tag(sensor_data)  #
        month, day, hour = gen_time_feature(sensor_data)

        self.train_tag = tag
        self.train_month, self.train_day, self.train_hour = month, day, hour  # 保存月份、日期、小时特征

        cos_d = cos_date(month, day, hour)  #  cos_d shape: (N_train, )
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(month, day, hour)  #  sin_d shape: (N_train, )
        sin_d = [[x] for x in sin_d]

        self.train_cos_d, self.train_sin_d = cos_d, sin_d  #

    def get_val_data_index(self, args):

        val_data = self.sensor_train_data_norm_list
        val_points = []  #

        near_len = args.output_len  #
        random.seed(args.val_seed)

        counts_val = 0

        while counts_val < args.val_size:  #


            i = random.randint(args.output_len, len(self.train_data) - self.data_lens - 1)  # self.train_data

            a1, a2 = 0, -13  #


            if (
                    (not np.isnan(self.sensor_train_data_norm_list[i: i + self.data_lens]).any())
                    and (
                        self.train_tag[i + args.input_len] <= a1  #
                        or a2 < self.train_tag[i + args.input_len] < 0  #
                        or 2 <= self.train_tag[i + args.input_len] <= 3  #
                    )
            ):
                # --------------------------------------------------

                self.train_tag[i + args.input_len] = 2  # tag 2 means in validation set

                for k in range(near_len):
                    self.train_tag[i + args.input_len - k] = 3  # tag 3 means near points of validation set
                    self.train_tag[i + args.input_len + k] = 3

                point = self.train_data_time[i + args.input_len]
                val_points.append([point])  # self.val_points

                counts_val = counts_val + 1  #


        self.val_points = val_points  #

        val_name = "%s" % (args.model)
        file_name = os.path.join(args.outf, val_name, "val", f"in{args.input_len}_out{args.output_len}_ro{args.roll}_validation_timestamps_24avg.tsv")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pd_temp = pd.DataFrame(data=val_points, columns=["Hold Out Start"])  # self.val_points
        pd_temp.to_csv(file_name, sep="\t")
        print("val set saved to : ", file_name)


    def get_train_data(self, args):
        """
        Get the training data as a DataLoader object.

        :param args: Arguments containing batch size and other parameters.
        :return: DataLoader object for training data.
        """
        train_data = self.sensor_train_data_norm_list

        DATA, DATAExm, DATANom = [], [], []  #  (train_days, feature_dim)
        Label, LabelExm, LableNom = [], [], []  #  (predict_days, label_dim)

        # randomly choose train data
        random.seed(args.train_seed)
        counts_normal = 0  #
        counts_oversamp = 0  #
        while counts_normal < args.train_volume:  #

            i = random.randint(args.output_len * 4, len(self.sensor_train_data_norm) - 31 * args.output_len * 4 - 1)


            pre1 = np.array(self.sensor_train_data_norm[(i + args.input_len): (i + args.input_len + args.output_len)])  # args.input_len: 输入窗口长度，args.output_len: 预测窗口长度; pre1 shape: (predict_days, )
            a1, a2 = 0, -13  #


            a3, max_index = None, None
            if np.max(pre1) > self.gm3_max_thres:
                a3 = args.os_s  # oversampling steps，
                max_index = np.argmax(pre1)
            elif np.min(pre1) < self.gm3_min_thres:
                a3 = args.os_s  # oversampling steps,
                max_index = np.argmin(pre1)
            a5 = args.os_v  #

            # ------------------------------------------------------------------
            # ------------------------------------------------------------------

            if (
                    (counts_oversamp < args.train_volume * (args.oversampling / 100))  #
                    and (np.max(pre1) > self.gm3_max_thres or np.min(pre1) < self.gm3_min_thres)  #
                    and (not np.isnan(self.sensor_train_data_norm_list[i: i + self.data_lens]).any())  #
                    and (self.train_tag[i + args.input_len] <= a1 or a2 < self.train_tag[i + args.input_len] < 0) #
            ):

                if a3 > 0:
                    i = i + max_index - 1  #
                    i = i - a3 * a5  #
                for kk in range(a3):
                    i = i + a5  #
                    if i > len(self.train_data) - 31 * args.output_len * 4 - 1 or i < args.output_len * 4:
                        continue  #

                    # ------------------------------------------------------
                    # ------------------------------------------------------
                    if (
                            not np.isnan(self.sensor_train_data_norm_list[i: i + self.data_lens]).any()
                            and self.train_tag[i + args.input_len] != 2
                            and self.train_tag[i + args.input_len] != 3
                            and self.train_tag[i + args.input_len] != 4
                    ):

                        data0 = np.array(self.sensor_train_data_norm_list[i: (i + args.input_len)]).reshape(args.input_len, -1)

                        # label0:   (dim 0)
                        label00 = np.array(self.sensor_train_data_norm[(i + args.input_len): (i + args.input_len + args.output_len)])
                        label0 = [[ff] for ff in label00]

                        #
                        b = i + args.input_len
                        e = i + args.input_len + args.output_len

                        # label2 / label3:
                        label2 = cos_date(self.train_month[b:e], self.train_day[b:e],
                                          self.train_hour[b:e])  # represent cos(int(data)) here
                        label2 = [[ff] for ff in label2]

                        label3 = sin_date(self.train_month[b:e], self.train_day[b:e],
                                          self.train_hour[b:e])  # represent sin(int(data)) here
                        label3 = [[ff] for ff in label3]

                        # label4:    (dim 3)
                        label4 = np.array(self.train_data[(i + args.input_len - 1):(i + args.input_len + args.output_len - 1)]).reshape(-1, 1)

                        # label5:      (dim 4)
                        label5 = np.array(self.train_data[(i + args.input_len): (i + args.input_len + args.output_len)]).reshape(-1, 1)

                        # label6: log_std normalization ground truth (dim 5)
                        label6 = np.array(self.log_std_train_data_norm[(i + args.input_len): (i + args.input_len + args.output_len)])  # log_std normalization ground truth
                        label6 = [[ff] for ff in label6]

                        # label7: dan_gmm3 past outlier score (dim 6)
                        label7 = np.array(self.R_sensor_data_norm[(i + args.input_len - args.output_len): (i + args.input_len)])
                        label7 = [[ff] for ff in label7]  #

                        # label8: dan_gmm3 current outlier score (dim 7)
                        label8 = np.array(self.R_sensor_data_norm[(i + args.input_len): (i + args.input_len + args.output_len)])  #
                        label8 = [[ff] for ff in label8]  #

                        #  (predict_days, 8)
                        label = np.concatenate((label0, label2), 1)
                        label = np.concatenate((label, label3), 1)
                        label = np.concatenate((label, label4), 1)
                        label = np.concatenate((label, label5), 1)
                        label = np.concatenate((label, label6), 1)  # dim=5 is log_std normalization ground truth
                        label = np.concatenate((label, label7), 1)  # dim=6 is dan_gmm3 past outlier score
                        label = np.concatenate((label, label8), 1)  # dim=7 is dan_gmm3 current outlier score

                        #
                        self.train_tag[i + args.input_len] = 4  # self.tag[i + self.train_days] = 4 means this point is sampled for training

                        counts_oversamp = counts_oversamp + 1  #

                        # 收集样本
                        DATA.append(data0)  # data0 shape (train_days, feature_dim)
                        Label.append(label)  # label shape (predict_days, 5), dim0: normed ground truth, dim1: cos date, dim2: sin date, dim3: pre-ground truth, dim4: ground truth

                        DATAExm.append(data0)
                        LabelExm.append(label)

            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            if ((not np.isnan(self.sensor_train_data_norm_list[i: i + self.data_lens]).any()) and  #
                    (self.train_tag[i + args.input_len] <= a1 or a2 < self.train_tag[i + args.input_len] < 0)):  #

                #
                data0 = np.array(self.sensor_train_data_norm_list[i: (i + args.input_len)]).reshape(args.input_len, -1)  #
                #
                label00 = np.array(self.sensor_train_data_norm[(i + args.input_len): (i + args.input_len + args.output_len)])  #

                # dim=0 is normed ground truth
                label0 = [[ff] for ff in label00]  #

                # cos / sin
                b = i + args.input_len  #
                e = i + args.input_len + args.output_len  #

                label2 = cos_date(self.train_month[b:e], self.train_day[b:e], self.train_hour[b:e])  # represent cos(int(data)) here, shape (predict_days, 1)
                label2 = [[ff] for ff in label2]

                label3 = sin_date(self.train_month[b:e], self.train_day[b:e], self.train_hour[b:e])  # represent sin(int(data)) here, shape (predict_days, 1)
                label3 = [[ff] for ff in label3]

                # dim=3 is pre-ground truth
                label4 = np.array(self.train_data[(i + args.input_len - 1):(i + args.input_len + args.output_len - 1)]).reshape(-1, 1)  #

                # dim=4 is ground truth
                label5 = np.array(self.train_data[(i + args.input_len): (i + args.input_len + args.output_len)]).reshape(-1, 1)  #

                # label6: log_std normalization ground truth (dim 5)
                label6 = np.array(self.log_std_train_data_norm[(i + args.input_len): (i + args.input_len + args.output_len)])  #
                label6 = [[ff] for ff in label6]

                # label7: dan_gmm3 past outlier score (dim 6)
                label7 = np.array(self.R_sensor_data_norm[(i + args.input_len - args.output_len): (i + args.input_len)])
                label7 = [[ff] for ff in label7]  #

                # label8: dan_gmm3 current outlier score (dim 7)
                label8 = np.array(self.R_sensor_data_norm[(i + args.input_len): (i + args.input_len + args.output_len)])  #
                label8 = [[ff] for ff in label8]  #


                # 拼接标签 final label shape: (predict_days, 8),
                label = np.concatenate((label0, label2), 1)
                label = np.concatenate((label, label3), 1)
                label = np.concatenate((label, label4), 1)
                label = np.concatenate((label, label5), 1)
                label = np.concatenate((label, label6), 1)  # dim=5 is log_std normalization ground truth
                label = np.concatenate((label, label7), 1)  # dim=6 is dan_gmm3 past outlier score
                label = np.concatenate((label, label8), 1)  # dim=7 is dan_gmm3 current outlier score
                # 每个维度表示：{dim0: normed ground truth, dim1: cos date, dim2: sin date, dim3: pre-ground truth, dim4: ground truth, dim5: log_std normalization ground truth, dim6: dan_gmm3 past outlier score, dim7: dan_gmm3 current outlier score}

                DATA.append(data0)
                Label.append(label)

                DATANom.append(data0)
                LableNom.append(label)

                self.train_tag[i + args.input_len] = 4
                counts_normal = counts_normal + 1

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        self.Train_DataSets = DATA

        # np.save('extre.npy', np.array(LabelExm, np.float32))
        # np.save('normal.npy', np.array(LableNom, np.float32))


        xx = np.array(self.Train_DataSets, np.float32)
        self.sample_wise_gmm.fit(np.squeeze(
            xx[:, -1 * args.gmm_l:, 1:2]))

        save_dir = os.path.join(args.outf, args.name)
        torch.save(self.sample_wise_gmm, os.path.join(save_dir, "sample_wise_GMM.pt"))

        self.gmm_means = np.squeeze(self.sample_wise_gmm.means_)
        print("time series gmm.weights are: ", self.sample_wise_gmm.weights_)


        gmm_prob30 = self.sample_wise_gmm.predict_proba(np.squeeze(np.array(self.Train_DataSets)[:, -1 * args.gmm_l:, 1:2]))  # gmm_prob30 是一个二维数组 (N_train, 3)，每行是一个样本在 3 个成分上的后验概率


        order1 = np.argmin(self.sample_wise_gmm.weights_)
        d0 = gmm_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmax(self.sample_wise_gmm.weights_)
        d1 = gmm_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("new order is, ", order1, order2, order3)
        d2 = gmm_prob30[:, order3].reshape(-1, 1)

        gmm_prob3 = np.concatenate((d0, d1), 1)
        gmm_prob3 = np.concatenate((gmm_prob3, d2), 1)

        # 每个样本把 (prob0, prob1, prob2) **沿时间维 repeat** 到 train_days
        prob0 = gmm_prob3[:, 0].reshape(-1, 1).repeat(args.input_len, axis=1)
        prob0 = prob0.reshape(len(prob0), -1, 1)  # (N_train, train_days, 1)
        prob1 = gmm_prob3[:, 1].reshape(-1, 1).repeat(args.input_len, axis=1)
        prob1 = prob1.reshape(len(prob1), -1, 1)  # (N_train, train_days, 1)
        prob2 = gmm_prob3[:, 2].reshape(-1, 1).repeat(args.input_len, axis=1)
        prob2 = prob2.reshape(len(prob2), -1, 1)  # (N_train, train_days, 1)
        prob = np.concatenate((prob0, prob1), 2)
        prob = np.concatenate((prob, prob2), 2)

        DATA = np.concatenate((DATA, prob), 2)  #
        print("DATA shape, ", np.array(self.Train_DataSets).shape)  #
        print("Label, ", np.array(Label).shape)

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------

        self.x_train = DATA
        self.y_train = np.array(Label)

        self.x_normal_train = DATANom
        self.y_normal_train = np.array(LableNom)

        train_data_tensor = RnnDataset(DATA, Label)
        self.train_data_loader = DataLoader(
            train_data_tensor,
            args.batchsize,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: x,
        )

    def refresh_dataset(self, args):

        all_train = self.all_input_data

        # read sensor data to vector
        start_num = all_train[all_train["datetime"] == args.train_start_point].index.values[0]
        print("for sensor ", args.reservoir_sensor, "start_num is: ", start_num)
        # foot label of train_end
        train_end = (all_train[all_train["datetime"] == args.train_end_point].index.values[0] - start_num)
        print("train set length is : ", train_end)

        k = all_train[all_train["datetime"] == args.test_end].index.values[0]
        self.sensor_all_data = all_train[start_num:k]

        # --------------------------------------------------
        # --------------------------------------------------
        self.all_data = np.array(self.sensor_all_data["value"].fillna(np.nan))
        self.all_data_time = np.array(self.sensor_all_data["datetime"].fillna(np.nan))

        self.sensor_all_data_norm = normalize_diff_with_stats(self.all_data, self.train_diff_mean, self.train_diff_std)  #
        self.sensor_all_data_norm_list = [[ff] for ff in self.sensor_all_data_norm]

        self.standnorm_all_data = standard_normalization_with_stats(self.all_data, self.stdn_mean, self.stdn_std)  #
        self.standnorm_all_data_list = [[ff] for ff in self.standnorm_all_data]

        self.logstdnorm_all_data = log_std_normalization_with_stats(self.all_data, self.logstd_mean, self.logstd_std)  #
        self.logstdnorm_all_data_list = [[ff] for ff in self.logstdnorm_all_data]

        self.ori_all_data_norm_list = [[ff] for ff in self.all_data]  #

        # --------------------------------------------------
        # --------------------------------------------------
        gmm_input = self.sensor_all_data_norm

        clean_data = []
        for ii in range(len(self.sensor_all_data_norm)):
            if (self.sensor_all_data_norm[ii] is not None) and (np.isnan(self.sensor_all_data_norm[ii]) != 1):
                clean_data.append(self.sensor_all_data_norm[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)

        data_prob3 = self.gm3.predict_proba(sensor_data_prob)  # (N,3)
        weights3 = self.gm3.weights_  #

        prob_in_distribution3 = (data_prob3[:, 0] * weights3[0] + data_prob3[:, 1] * weights3[1] + data_prob3[:, 2] * weights3[2])

        prob_like_outlier3 = 1 - prob_in_distribution3  # (N,)
        prob_like_outlier3 = prob_like_outlier3.reshape((len(sensor_data_prob), 1))

        recover_data = []
        temp = np.zeros(np.array(data_prob3[0]).shape)
        jj = 0
        for ii in range(len(self.sensor_all_data_norm)):
            if (self.sensor_all_data_norm[ii] is not None) and (
                    np.isnan(self.sensor_all_data_norm[ii]) != 1
            ):
                recover_data.append(prob_like_outlier3[jj])  #
                jj = jj + 1
            else:
                recover_data.append(self.sensor_all_data_norm[ii])
        prob_like_outlier3 = np.array(recover_data, np.float32).reshape(len(self.sensor_all_data_norm), 1)

        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, prob_like_outlier3), 1)  #

        clean_data = []
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                clean_data.append(gmm_input[ii])
        sensor_all_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)

        self.gmm0_means = np.squeeze(self.gmm0.means_)
        weights3 = self.gmm0.weights_
        all_data_prob30 = self.gmm0.predict_proba(sensor_all_data_prob)

        order1 = np.argmax(weights3)
        d0 = all_data_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmin(weights3)
        d1 = all_data_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("new order is, ", order1, order2, order3)

        data_prob3 = np.concatenate((d0, d1), 1)
        data_prob3 = np.concatenate((data_prob3, all_data_prob30[:, order3].reshape(-1, 1)), 1)

        recover_prob = []
        temp = np.zeros(np.array(data_prob3[0]).shape)
        jj = 0
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                recover_prob.append(data_prob3[jj])
                jj = jj + 1
            else:
                recover_prob.append(temp)
        recover_prob = np.array(recover_prob, np.float32).reshape(len(gmm_input), -1)

        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, recover_prob[:, 0:1]), 1)
        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, recover_prob[:, 1:2]), 1)
        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, recover_prob[:, 2:3]), 1)


        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, self.standnorm_all_data_list), 1)
        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, self.ori_all_data_norm_list), 1)

        all_data_time_str = self.all_data_time.astype(str)
        all_data_time_pd = pd.to_datetime(all_data_time_str)

        year = all_data_time_pd.year
        month = all_data_time_pd.month
        day = all_data_time_pd.day
        hour = all_data_time_pd.hour
        minute = all_data_time_pd.minute

        all_time_features = np.stack([year, month, day, hour, minute], axis=1)

        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, all_time_features), 1)

        # dim 12  log_std normalization
        clean_data = []
        for ii in range(len(self.all_data)):
            if (self.all_data[ii] is not None) and (np.isnan(self.all_data[ii]) != 1):
                clean_data.append(self.all_data[ii])
        sensor_data_prob = np.array(clean_data).reshape(-1, 1)

        dan_weights3 = self.dangmm3.weights_
        dan_data_prob3 = self.dangmm3.predict_proba(sensor_data_prob)
        dan_prob_in_distribution3 = (
                dan_data_prob3[:, 0] * dan_weights3[0]
                + dan_data_prob3[:, 1] * dan_weights3[1]
                + dan_data_prob3[:, 2] * dan_weights3[2]
        )

        dan_prob_like_outlier3 = 1 - dan_prob_in_distribution3
        dan_prob_like_outlier3 = dan_prob_like_outlier3.reshape(len(sensor_data_prob), 1)

        dan_recover_data = []
        temp = 0
        jj = 0
        for ii in range(len(self.all_data)):
            if (self.all_data[ii] is not None) and (np.isnan(self.all_data[ii]) != 1):
                dan_recover_data.append(dan_prob_like_outlier3[jj])
                jj = jj + 1
            else:
                dan_recover_data.append(self.all_data[ii])
        dan_all_prob_like_outlier3 = np.array(dan_recover_data, np.float32).reshape(len(self.all_data), 1)

        # self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, dan_all_prob_like_outlier3), 1)
        self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, self.logstdnorm_all_data_list), 1)

        # Lable
        self.R_all_data = dan_all_prob_like_outlier3
        self.R_all_sensor_data_norm = log_std_normalization_with_stats(self.R_data, self.R_mean, self.R_std)

        # self.R_all_sensor_data_norm1 = dan_all_prob_like_outlier3.squeeze()   # before
        self.R_all_sensor_data_norm1 = self.R_all_sensor_data_norm.squeeze()
        self.R_all_sensor_data_norm = self.R_sensor_data_norm1  # shape (len(self.data),)

        print("Finish prob indicator updating.")


        self.all_tag = gen_month_tag(self.sensor_all_data)
        self.all_month, self.all_day, self.all_hour = gen_time_feature(self.sensor_all_data)  # update

        cos_d = cos_date(self.all_month, self.all_day, self.all_hour)
        self.all_cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.all_month, self.all_day, self.all_hour)
        self.all_sin_d = [[x] for x in sin_d]

        # self.sensor_all_data_norm_list = np.concatenate((self.sensor_all_data_norm_list, cos_d, sin_d), axis=1)

    def get_test_data_index(self, args):
        """
        Get the test data as a DataLoader object.

        :param args: Arguments containing batch size and other parameters.
        :return: DataLoader object for test data.
        """

        test_points = []
        self.refresh_dataset(args)


        start_num = self.all_input_data[self.all_input_data["datetime"] == args.train_start_point].index.values[0]


        begin_num = (self.all_input_data[self.all_input_data["datetime"] == args.test_start].index.values[0] - start_num)

        end_num = (self.all_input_data[self.all_input_data["datetime"] == args.test_end].index.values[0] - start_num)


        iterval = args.roll

        for i in range(int((end_num - begin_num - args.output_len) / iterval)):  # do inference every 24 hours
            point = self.all_data_time[begin_num + i * iterval]

            if not np.isnan(
                    np.array(
                        self.all_data[begin_num + i * iterval - args.input_len: begin_num + i * iterval + args.output_len] # 窗口的左端，窗口的右端；
                    )
            ).any():
                test_points.append([point])


        self.test_data_index = test_points
        # ------------------------------------------------------------------
        print("Finish getting test data")

    def get_batch_data(self, time_point_list, args):
        results = []
        all_data = self.sensor_all_data.sort_values("datetime").copy()
        all_data = all_data.reset_index(drop=True)

        for t in tqdm(time_point_list, desc="Processing time points"):
            if isinstance(t, list):
                time_point = t[0]
            else:
                time_point = t
            result = self.get_single_data(time_point, all_data, args)
            if result:
                results.append(result)

        if not results:
            return (
                np.empty((0, args.input_len, 10)),  # x_tests
                np.empty((0, args.output_len, 5)),  # y_tests
            )

        x_tests = []
        logstd_gts = []
        dan_now_outliers = []
        dan_pre_outliers = []
        norm_gts = []
        ts_features = []
        pre_gts = []
        gts = []

        for x_test, logstd_norm_y_test, dan_now_y_prob_like_outlier3, dan_pre_y_prob_like_outlier3, norm_gt, ts_f, pre_gt, gt in results:
            x_tests.append(x_test)  # shape (1, 360, 10)
            norm_gts.append(np.expand_dims(norm_gt, axis=0))  # shape (1, 72, 1)
            ts_features.append(ts_f)  # shape (1, 72, 2)
            pre_gts.append(pre_gt)  # scalar
            gts.append(np.expand_dims(gt.reshape(72, 1), axis=0))  # shape (1, 72, 1)
            # DAN
            logstd_gts.append(np.expand_dims(logstd_norm_y_test, axis=0))   # shape (1, 72, 1)
            dan_now_outliers.append(np.expand_dims(dan_now_y_prob_like_outlier3, axis=0))   # shape (1, 72, 1)
            dan_pre_outliers.append(np.expand_dims(dan_pre_y_prob_like_outlier3, axis=0))   # shape (1, 72, 1)

        x_tests_np = np.concatenate(x_tests, axis=0)  # (N, 360, 16)
        norm_gts_np = np.concatenate(norm_gts, axis=0)  # (N, 72, 1)
        ts_features_np = np.concatenate(ts_features, axis=0)  # (N, 72, 2)
        pre_gts_np = np.repeat(np.array(pre_gts)[:, None], args.output_len, axis=1)[..., None]   # (N, 72, 1)
        gts_np = np.concatenate(gts, axis=0)  # (N, 72, 1)
        # DAN
        logstd_gts_np = np.concatenate(logstd_gts, axis=0)
        dan_now_outliers_np = np.concatenate(dan_now_outliers, axis=0)
        dan_pre_outliers_np = np.concatenate(dan_pre_outliers, axis=0)

        y_tests_np = np.concatenate(
            [norm_gts_np, ts_features_np, pre_gts_np, gts_np, logstd_gts_np, dan_pre_outliers_np, dan_now_outliers_np],
            axis=-1
        )

        return x_tests_np, y_tests_np


    def get_single_data(self, time_point, all_data, args):

        try:
            point = all_data[all_data["datetime"] == time_point].index.values[0]
        except IndexError:
            print(f"Time point {time_point} not found in data.")
            return None

        iloc_point = all_data.index.get_loc(point)
        if iloc_point + args.output_len > len(all_data) or iloc_point < args.input_len:
            print(f"Time point {time_point} is out of valid range.")
            return None

        reservoir_data = all_data[point - args.input_len: point]["value"].values.tolist()
        pre_gt = np.array(all_data[point - 1: point]["value"])
        pre_gt = pre_gt[0]
        gt = np.array(all_data[point: point + args.output_len]["value"])

        if pre_gt is None:
            print("pre_gt is None, please fill it or switch to another time point.")
        NN = np.isnan(reservoir_data).any()
        if NN:
            print("There is None value in the input sequence.")

        test_month = []
        test_day = []
        test_hour = []
        test_year = []
        test_minute = []

        new_time = datetime.strptime(time_point, "%Y-%m-%d %H:%M:%S")
        for i in range(args.output_len):
            new_time_temp = new_time + timedelta(minutes=30)
            new_time = new_time.strftime("%Y-%m-%d %H:%M:%S")

            year = int(new_time[0:4])
            month = int(new_time[5:7])
            day = int(new_time[8:10])
            hour = int(new_time[11:13])
            minute = int(new_time[14:16])

            test_month.append(month)
            test_day.append(day)
            test_hour.append(hour)
            test_year.append(year)
            test_minute.append(minute)

            new_time = new_time_temp

        y2 = cos_date(test_month, test_day, test_hour)
        y2 = [[ff] for ff in y2]

        y3 = sin_date(test_month, test_day, test_hour)
        y3 = [[ff] for ff in y3]

        test_ts_features = np.array([np.concatenate((y2, y3), 1)])
        test_timestamp_features = np.array([np.stack([test_year, test_month, test_day, test_hour, test_minute], axis=1)])

        x_test = np.array(normalize_diff_with_stats(reservoir_data, self.train_diff_mean, self.train_diff_std), np.float32).reshape(args.input_len, -1)
        norm_y_test = np.array(normalize_diff_with_stats(all_data[point: point + args.output_len]["value"].values.tolist(), self.train_diff_mean, self.train_diff_std), np.float32).reshape(args.output_len, -1)

        gmm_input = x_test
        weights3 = self.gm3.weights_
        data_prob3 = self.gm3.predict_proba(np.array(x_test)[:, 0:1].reshape(-1, 1))
        prob_in_distribution3 = (
                data_prob3[:, 0] * weights3[0]
                + data_prob3[:, 1] * weights3[1]
                + data_prob3[:, 2] * weights3[2]
        )
        prob_like_outlier3 = 1 - prob_in_distribution3
        prob_like_outlier3 = prob_like_outlier3.reshape(-1, 1)
        prob_like_outlier3 = np.array(prob_like_outlier3, np.float32).reshape(-1, 1)
        x_test = np.concatenate((x_test, prob_like_outlier3), 1)

        self.gmm0_means = np.squeeze(self.gmm0.means_)
        weights3 = self.gmm0.weights_
        data_prob30 = self.gmm0.predict_proba(np.array(gmm_input)[:, 0:1].reshape(-1, 1))
        order1 = np.argmax(weights3)
        d0 = data_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmin(weights3)
        d1 = data_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        data_prob3 = np.concatenate((d0, d1), 1)
        data_prob3 = np.concatenate((data_prob3, data_prob30[:, order3].reshape(-1, 1)), 1)
        recover_prob = np.array(data_prob3, np.float32)
        x_test = np.concatenate((x_test, recover_prob[:, 0:1]), 1)
        x_test = np.concatenate((x_test, recover_prob[:, 1:2]), 1)
        x_test = np.concatenate((x_test, recover_prob[:, 2:3]), 1)

        stdn_x_test = np.array(standard_normalization_with_stats(reservoir_data, self.stdn_mean, self.stdn_std), np.float32).reshape(args.input_len, -1)
        ori_x_test = np.array(reservoir_data, np.float32).reshape(args.input_len, -1)

        x_test = np.concatenate((x_test, stdn_x_test), 1)  # 拼接标准化数据
        x_test = np.concatenate((x_test, ori_x_test), 1)

        x_timestamp = all_data[point - args.input_len: point]["datetime"].values
        data_time_str = x_timestamp.astype(str)
        data_time_pd = pd.to_datetime(data_time_str)

        year = data_time_pd.year
        month = data_time_pd.month
        day = data_time_pd.day
        hour = data_time_pd.hour
        minute = data_time_pd.minute

        x_time_features = np.stack([year, month, day, hour, minute], axis=1)
        x_test = np.concatenate((x_test, x_time_features), 1)

        # input dimension 13
        logstd_x_test = np.array(log_std_normalization_with_stats(reservoir_data, self.logstd_mean, self.logstd_std), np.float32).reshape(args.input_len, -1)

        dan_x_weights3 = self.dangmm3.weights_
        dan_x_data_prob3 = self.dangmm3.predict_proba(np.array(reservoir_data).reshape(-1, 1))
        dan_x_prob_in_distribution3 = (
                dan_x_data_prob3[:, 0] * dan_x_weights3[0]
                + dan_x_data_prob3[:, 1] * dan_x_weights3[1]
                + dan_x_data_prob3[:, 2] * dan_x_weights3[2]
        )

        dan_x_prob_like_outlier3 = 1 - dan_x_prob_in_distribution3
        dan_x_prob_like_outlier3 = dan_x_prob_like_outlier3.reshape(-1, 1)
        dan_x_prob_like_outlier3 = np.array(dan_x_prob_like_outlier3, np.float32).reshape(-1, 1)

        dan_x_prob_like_outlier3 = np.array(log_std_normalization_with_stats(dan_x_prob_like_outlier3, self.R_mean, self.R_std)).reshape(args.input_len, -1)

        # x_test = np.concatenate((x_test, dan_x_prob_like_outlier3), 1)
        x_test = np.concatenate((x_test, logstd_x_test), 1)  # 拼接 log_std normalization 到 dim13

        # input dimension  14, 15, 16
        gmm_prob30 = self.sample_wise_gmm.predict_proba(np.array(x_test)[-1 * args.gmm_l:, 1:2].reshape(1, -1))
        order1 = np.argmin(self.sample_wise_gmm.weights_)
        d0 = gmm_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmax(self.sample_wise_gmm.weights_)
        d1 = gmm_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        d2 = gmm_prob30[:, order3].reshape(-1, 1)
        gmm_prob3 = np.concatenate((d0, d1), 1)
        gmm_prob3 = np.concatenate((gmm_prob3, d2), 1)
        prob0 = gmm_prob3[:, 0].reshape(-1, 1).repeat(args.input_len, axis=1)
        prob0 = prob0.reshape(len(prob0), -1, 1)
        prob1 = gmm_prob3[:, 1].reshape(-1, 1).repeat(args.input_len, axis=1)
        prob1 = prob1.reshape(len(prob1), -1, 1)
        prob2 = gmm_prob3[:, 2].reshape(-1, 1).repeat(args.input_len, axis=1)
        prob2 = prob2.reshape(len(prob2), -1, 1)
        prob = np.concatenate((prob0, prob1), 2)
        prob = np.concatenate((prob, prob2), 2)
        x_test = [x_test]
        x_test = np.concatenate((x_test, prob), 2)


        # lable dim 5
        logstd_norm_y_test = np.array(log_std_normalization_with_stats(all_data[point: point + args.output_len]["value"].values.tolist(),
                                      self.logstd_mean, self.logstd_std), np.float32).reshape(args.output_len, -1)

        # lable dimension 6 past outlier score
        dan_pre_gmm3_inputs = np.array(reservoir_data, np.float32).reshape(-1, 1)[-1 * args.output_len:]    # shape (output_len, 1)
        dan_pre_y_weights3 = self.dangmm3.weights_
        dan_pre_y_data_prob3 = self.dangmm3.predict_proba(dan_pre_gmm3_inputs)
        dan_pre_y_prob_in_distribution3 = (
                dan_pre_y_data_prob3[:, 0] * dan_pre_y_weights3[0]
                + dan_pre_y_data_prob3[:, 1] * dan_pre_y_weights3[1]
                + dan_pre_y_data_prob3[:, 2] * dan_pre_y_weights3[2]
        )

        dan_pre_y_prob_like_outlier3 = 1 - dan_pre_y_prob_in_distribution3
        dan_pre_y_prob_like_outlier3 = dan_pre_y_prob_like_outlier3.reshape(-1, 1)
        dan_pre_y_prob_like_outlier3 = np.array(log_std_normalization_with_stats(dan_pre_y_prob_like_outlier3, self.R_mean, self.R_std)).reshape(args.output_len, -1)

        # lable dim 7
        gt_list = np.array(all_data[point: point + args.output_len]["value"].values.tolist())
        dan_now_gmm3_inputs = np.array(gt_list, np.float32).reshape(-1, 1)

        dan_now_y_weights3 = self.dangmm3.weights_
        dan_now_y_data_prob3 = self.dangmm3.predict_proba(dan_now_gmm3_inputs)
        dan_now_y_prob_in_distribution3 = (
                dan_now_y_data_prob3[:, 0] * dan_now_y_weights3[0]
                + dan_now_y_data_prob3[:, 1] * dan_now_y_weights3[1]
                + dan_now_y_data_prob3[:, 2] * dan_now_y_weights3[2]
        )

        dan_now_y_prob_like_outlier3 = 1 - dan_now_y_prob_in_distribution3
        dan_now_y_prob_like_outlier3 = dan_now_y_prob_like_outlier3.reshape(-1, 1)
        dan_now_y_prob_like_outlier3 = np.array(log_std_normalization_with_stats(dan_now_y_prob_like_outlier3, self.R_mean, self.R_std)).reshape(args.output_len, -1)


        return x_test, logstd_norm_y_test, dan_now_y_prob_like_outlier3, dan_pre_y_prob_like_outlier3, norm_y_test, test_ts_features, pre_gt, gt



def data_generation(task_name: str, arg_file_path: str = None):
    parser = argparse.ArgumentParser(description=task_name)

    # default settings

    # dataset and sampling parameters
    parser.add_argument("--data_path", type=str, default="../datasets/", help="path to the reservoir dataset")
    parser.add_argument("--reservoir_sensor", default="reservoir_stor_4007_sof24", help="reservoir dataset", )
    parser.add_argument("--name", type=str, default="test", help="name of the experiment")
    parser.add_argument("--rain_sensor", type=str, default="rain_sensor", help="rain sensor name")

    parser.add_argument("--train_seed", type=int, default=1010, help="random seed for train sampling")
    parser.add_argument("--test_seed", type=int, default=2000, help="random seed for test sampling")
    parser.add_argument("--val_seed", type=int, default=2007, help="random seed for val sampling")
    parser.add_argument("--train_volume", type=int, default=20000, help="train set size")
    parser.add_argument("--val_size", type=int, default=60, help="validation set size")

    parser.add_argument("--train_start_point", type=str, default="1991-07-01 23:30:00",
                        help="start time of the train set", )
    parser.add_argument("--train_end_point", type=str, default="2018-06-30 23:30:00",
                        help="end time of the train set", )
    parser.add_argument("--test_start", type=str, default="2018-07-01 00:30:00", help="start time of the test set", )
    parser.add_argument("--test_end", type=str, default="2019-07-01 00:30:00", help="end time of the test set", )

    parser.add_argument("--oversampling", type=int, default=30, help="ratio of training data with extreme points.", )
    parser.add_argument("--os_s", type=int, default=1, help="oversampling steps")  # 默认是0
    parser.add_argument("--os_v", type=int, default=5, help="oversampling frequency")

    # GMM parameters
    parser.add_argument("--seq_weight", type=float, default=0.3, help="sequence cluster weight")
    # parser.add_argument("--gmm_save_path", type=str, default="./", help="path to save the GMM model")

    # input and output parameters
    parser.add_argument("--input_dim", type=int, default=1, help="input dimension")
    parser.add_argument("--output_dim", type=int, default=1, help="output dimension")
    parser.add_argument("--input_len", type=int, default=15 * 24, help="length of input vector")
    parser.add_argument("--output_len", type=int, default=24 * 3, help="length of output vector")
    parser.add_argument("--roll", type=int, default=8, help="roll step for inference")

    # model parameters
    parser.add_argument("--hidden_dim", type=int, default=512, help="hidden dim of basic layers")
    parser.add_argument("--atten_dim", type=int, default=300, help="hidden dim of attention layers")
    parser.add_argument("--layer", type=int, default=2, help="number of layers")
    parser.add_argument("--model", type=str, default="4009", help="model label")

    # training parameters
    parser.add_argument("--batchsize", type=int, default=48, help="batch size of train data")
    parser.add_argument("--epochs", type=int, default=1, help="train epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lradj", type=str, default="type4", help="learning rate adjustment policy")
    parser.add_argument("--mode", type=str, default="train", help="set it to train or inference with an existing pt_file", )
    parser.add_argument("--arg_file", type=str, default="", help=".txt file. If set, reset the default parameters defined in this file.", )
    parser.add_argument("--save", type=int, default=0, help="1 if save the predicted file of testset, else 0", )
    parser.add_argument("--outf", default="./", help="output folder")
    parser.add_argument('--use_gpu', default=False, help='use gpu or not')

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu ids: e.g. 0. use -1 for CPU")
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--watershed", type=int, default=0, help="watershed index")

    cli_args = []

    file_args = []
    if arg_file_path and os.path.isfile(arg_file_path):
        file_args = parse_kv_argfile(arg_file_path)

    args = parser.parse_args(file_args + cli_args)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.gmm_l = args.output_len

    init_data = DataGenerate(args.data_path, args)

    print(f"Initializing data generation for position:{args.name}!")

if __name__ == '__main__':
    # initial_seed(2025)

    positions = ['Almaden', 'Coyote', 'Lexington', 'Stevens_Creek', 'Vasona']

    for pos in positions:
        config_path = f"../records/mcann_configs/{pos}.txt"
        data_generation('test', arg_file_path=config_path)
