#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn

from abc import abstractmethod

class Scale(object):
    """ Do nothing. """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x):
        return self

    @abstractmethod
    def transform(self, x):
        return x

    @abstractmethod
    def inverse_transform(self, x, type=None, part='train', obj='predict'):
        return x


class StandardNorm(Scale):

    """ Standard normalization (Z-score). """

    def __init__(self, agrs, mean=None, std=None,
                 diff_mean=None, diff_std=None,
                 diff_train_pre_one_step_x=None,
                 diff_test_pre_one_step_x=None,
                 diff_val_pre_one_step_x=None):
        """
        Standard normalization (Z-score) scaler.
        :param mean: pre-computed mean value for normalization.
        :param std: pre-computed standard deviation value for normalization.
        """
        super(StandardNorm, self).__init__()

        self.mean = mean
        self.std = std
        self.diff_mean = diff_mean
        self.diff_std = diff_std
        self.diff_train_pre_one_step_x = diff_train_pre_one_step_x
        self.diff_test_pre_one_step_x = diff_test_pre_one_step_x
        self.diff_val_pre_one_step_x = diff_val_pre_one_step_x

        self.diff_real_train_y = None
        self.diff_real_test_y = None
        self.diff_real_val_y = None

    def fit(self, x):
        # """ Fit the scaler to the data. """
        # self.mean = np.mean(x)
        # self.std = np.std(x)
        pass
        return self

    def transform(self, x):
        """ Transform the data using the fitted scaler. """
        return (x - self.mean) / self.std

    def inverse_transform(self, x, type=None, part='train', obj='predict'):
        """ Inverse transform the data back to original scale. """
        if type == 'ori' or type == 'std':
            return x * self.std + self.mean



