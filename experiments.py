#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import argparse

from util.scale import Scale
from util.parse import parse_resolution


parser = argparse.ArgumentParser()

# input and output parameters
parser.add_argument("--input_dim", type=int, default=1, help="input dimension")
parser.add_argument("--output_dim", type=int, default=1, help="output dimension")
parser.add_argument("--input_len", type=int, default=15 * 24, help="length of input vector")
parser.add_argument("--output_len", type=int, default=24 * 3, help="length of output vector")
parser.add_argument("--roll", type=int, default=8, help="roll step for inference")
parser.add_argument("--model", type=str, default="4009", help="model label")

# training parameters
parser.add_argument('--status', type=str, default='train', help='status of the model, train or test')
parser.add_argument('--batchstep', type=int, default=0, help='batch size for test step, 0 for no test step')
parser.add_argument("--batchsize", type=int, default=48, help="batch size of train data")
parser.add_argument("--epochs", type=int, default=60, help="train epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--lradj", type=str, default="type4", help="learning rate adjustment policy")
parser.add_argument('--use_norm', default=True, help='use normalization or not')
parser.add_argument('--norm_type', type=str, default='std',
                    help='norm type for data')  # 'std' for standardization, 'diff' for difference normalization, 'ori' for original data, None for no normalization

# environment parameters
parser.add_argument("--mode", type=str, default="train", help="set it to train or inference with an existing pt_file", )
parser.add_argument("--arg_file", type=str, default="",
                    help=".txt file. If set, reset the default parameters defined in this file.", )
parser.add_argument("--save", type=int, default=0, help="1 if save the predicted file of testset, else 0", )
parser.add_argument("--outf", default="./", help="output folder")
parser.add_argument('--use_gpu', default=False, help='use gpu or not')
parser.add_argument('--cwt_dir', type=str, default='./cwt_features/', help='directory for cwt data')

parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument("--watershed", type=int, default=0, help="watershed index")
parser.add_argument("--data_set", type=str, default="Coyote", help="dataset name")
# positions = ['Almaden', 'Coyote', 'Lexington', 'Stevens_Creek', 'Vasona']

# model_parameters
parser.add_argument("--dropout_rate", type=float, default=0., help="dropout rate for the model")
parser.add_argument("--expert_num", type=int, default=3, help="number of experts in the model")
parser.add_argument("--num_scales", type=int, default=16, help="number of scales in the model")
parser.add_argument("--resolution_size", type=parse_resolution, default=[12, 24], help="resolution size for the model (comma-separated, e.g. '12,24')")
parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension for the model")
parser.add_argument("--gmm_hidden_dim", type=int, default=8, help="hidden dimension for GMM feature in the model")
parser.add_argument("--ts_ratio", type=float, default=0.1, help="time series ratio for the model")
parser.add_argument("--moe_hidden_dim", type=int, default=1, help="hidden dimension for MoE in the model")
parser.add_argument("--fs", type=float, default=1.0, help="sampling frequency for the model")

parser.add_argument("--early_stop_threshold", type=int, default=-1, help="early stop threshold for the model")
parser.add_argument("--align_loss_ratio", type=float, default=0.05, help="ratio for align loss in the model")
parser.add_argument("--diversity_loss_ratio", type=float, default=0.05, help="ratio for diversity loss in the model")
parser.add_argument("--cuda_devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES, e.g., '0,1,2'")  # 新增参数
parser.add_argument("--comments", type=str, default="",)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import warnings
warnings.filterwarnings("ignore", message="Complex modules are a new feature")

import pandas as pd
import numpy as np
import random
import pickle

import torch
print(f"[DEBUG] Torch sees {torch.cuda.device_count()} GPUs")

import torch.nn as nn
from torch.utils.data import DataLoader

from util.data import initial_seed
from util.tools import get_data, get_statistical, get_cwt_tensor, get_time_feature
from util.trainer import Trainer, model_predictor
from util.scale import StandardNorm

from model.M2FMoE import M2FMoE

def main():

    args.float_type = torch.float32
    args.status = 'train'
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = f"processed_datasets/output/{args.data_set}/in{args.input_len}_out{args.output_len}_ro{args.roll}"

    train_diff_mean, train_diff_std, train_min, train_mean, train_std = get_statistical(file_path)

    Scale = None
    if args.norm_type == 'ori':
        Scale = None
    elif args.norm_type == 'std':
        Scale = StandardNorm(agrs=args, mean=train_mean, std=train_std)

    train_x, train_y, test_x, test_y, val_x, val_y = None, None, None, None, None, None

    # 'ori' for original data, 'std' for standardized data
    if args.norm_type == 'ori':
        train_x, train_y, test_x, test_y, val_x, val_y = get_data(args, file_path,'ori', Scale)
    elif args.norm_type == 'std':
        train_x, train_y, test_x, test_y, val_x, val_y = get_data(args, file_path, 'std', Scale)

    # cwt_train_x, cwt_val_x, cwt_test_x, feq_train_x, feq_val_x, feq_test_x = get_cwt_tensor(train_x, val_x, args, test_x)
    cwt_train_x, cwt_val_x, feq_train_x, feq_val_x = get_cwt_tensor(train_x, val_x, args)

    if args.norm_type == 'ori':
        args.use_norm = True
    else:
        args.use_norm = False

    model = M2FMoE(train_x.shape[1], train_x.shape[2], train_y.shape[1], dec_in=train_y.shape[2], dropout_rate=args.dropout_rate, expert_num=args.expert_num,
                    num_scales=args.num_scales, resolution_size=args.resolution_size, use_revin=args.use_norm, hidden_dim=args.hidden_dim, moe_hidden_dim=args.moe_hidden_dim, ts_ratio=args.ts_ratio)

    model_name = type(model).__name__
    args.model_name = model_name
    model = model.to(args.device, dtype=args.float_type)

    print(model_name)
    print(train_x.shape, train_y.shape, '\n', val_x.shape, val_y.shape, '\n', test_x.shape, test_y.shape)

    print('seq_len:{}, pre_len:{}, num_scales:{}, expert_num:{}, resolution_size:{}, fs:{}, use_revin:{}, hidden_dim:{}, '
          'moe_hidd_dim:{}, ts_ratio:{}, dropout_rate:{}'.format(
            train_x.shape[1], train_y.shape[1], args.num_scales, args.expert_num, [1] + args.resolution_size, 1, args.use_norm,
            args.hidden_dim, args.moe_hidden_dim, args.ts_ratio, args.dropout_rate
        ))

    assert args.mode in ['train', 'test']
    if args.mode == 'test':
        op_model_path = f"./save_model/{model_name}/{args.data_set}/pl{args.output_len}_in{args.input_len}_ro{args.roll}_ns{args.norm_type}.pkl"
        with open(op_model_path, "rb") as f:
            saved_data = pickle.load(f)
            model.load_state_dict(saved_data["model_state_dict"])

    elif Scale is not None and (args.norm_type == 'std'):
        op_model = Trainer(model, [train_x, cwt_train_x, feq_train_x, train_y],
                           validation_data=[val_x, cwt_val_x, feq_val_x, val_y],
                           shuffle=False, verbose=True, args=args,
                           early_stop_threshold=args.early_stop_threshold,
                           scaler=Scale, save_model=True,
                           criterion=nn.MSELoss(reduction="sum") )

    args.status = 'test'
    cwt_test_x, feq_test_x = get_cwt_tensor(test_x, None, args)

    y_test = test_y
    y_test_hat = model_predictor(model, [test_x, cwt_test_x, feq_test_x], args)

    y_test = Scale.inverse_transform(y_test, args.norm_type, 'test', 'real')
    y_test_hat = Scale.inverse_transform(y_test_hat, args.norm_type, 'test', 'predict')

    y_test_np = y_test.squeeze(-1).detach().cpu().numpy()
    y_test_hat_np = y_test_hat.squeeze(-1).detach().cpu().numpy()

    np.save(f'./predictions/{args.model_name}_{args.data_set}_y_test.npy', y_test_np)
    np.save(f'./predictions/{args.model_name}_{args.data_set}_y_test_hat.npy', y_test_hat_np)


if __name__ == '__main__':
    initial_seed(1010)

    main()