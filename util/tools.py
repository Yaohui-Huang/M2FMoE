#!/usr/bin/env python
# encoding: utf-8
import os
import numpy as np
import random
import pywt
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import math
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .time_features import time_features


def to_string(*kwargs):
    """Several numbers to string."""
    _list = [str(kwargs[0])] + [str(_t) if isinstance(_t, str) else '{:.5f}'.format(_t) for _t in kwargs[1:]]
    total = ' \t'.join(_list)  # join these strings to another string
    return total


def adjust_learning_rate(optimizer, epoch, args):


    if args.lradj == "type1":       # type1
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":     # type2
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":     # type3
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == "type4":     # type4
        lr_adjust = {epoch: args.learning_rate * (0.98 ** ((epoch - 1) // 1))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def model_predictor(model: nn.Module, inputs: [tuple or list], args):
    """
        :param model: the model instance.
        :param inputs: input tensors (a.k.a., X).
        :param args: arguments containing batch size and other configurations.
        :return: prediction tensor
    """
    outputs = []
    dataset = data.TensorDataset(*inputs)

    if args.status == 'test':
        loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        loader = data.DataLoader(dataset=dataset, batch_size=args.batchsize, shuffle=False)

    for step, batch_inputs in enumerate(loader):

        args.batchstep = step
        if hasattr(model, 'args'):
            model.args = args

        out = model(*batch_inputs)
        if isinstance(out, (tuple, list)):
            out = out[0]

        out = out.detach()

        outputs.append(out)
    prediction = torch.cat(outputs, dim=0)
    return prediction



def model_predictor_test(model: nn.Module, inputs: [tuple or list], args):
    """
        :param model: the model instance.
        :param inputs: input tensors (a.k.a., X).
        :param args: arguments containing batch size and other configurations.
        :return: prediction tensor
    """
    outputs = []
    dataset = data.TensorDataset(*inputs)

    if args.status == 'test':
        loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        loader = data.DataLoader(dataset=dataset, batch_size=args.batchsize, shuffle=False)

    cwtw_list = []
    fftw_list = []
    for step, batch_inputs in enumerate(loader):

        args.batchstep = step
        if hasattr(model, 'args'):
            model.args = args

        outres = model(*batch_inputs)
        if isinstance(outres, (tuple, list)):
            out = outres[0]

        out = out.detach()

        outputs.append(out)
        cwtw_list.append(outres[-2].detach())
        fftw_list.append(outres[-1].detach())

    prediction = torch.cat(outputs, dim=0)
    cwtw = torch.cat(cwtw_list, dim=0)
    fftw = torch.cat(fftw_list, dim=0)

    return prediction, cwtw, fftw


def get_data(args, file_path, type='all', Scale=None):
    """Load and preprocess data for training and testing."""

    test_x = np.load(os.path.join(file_path, "test_x.npy"))
    test_y = np.load(os.path.join(file_path, "test_y.npy"))
    train_x = np.load(os.path.join(file_path, "train_x.npy"))
    train_y = np.load(os.path.join(file_path, "train_y.npy"))
    val_x = np.load(os.path.join(file_path, "val_x.npy"))
    val_y = np.load(os.path.join(file_path, "val_y.npy"))

    assert type in ['all', 'ori', 'std']

    train_x_tensor, train_y_tensor, test_x_tensor, test_y_tensor, val_x_tensor, val_y_tensor = None, None, None, None, None, None

    if type == 'all':
        train_x_tensor = torch.tensor(train_x, dtype=args.float_type).to(args.device)
        train_y_tensor = torch.tensor(train_y, dtype=args.float_type).to(args.device)
        test_x_tensor = torch.tensor(test_x, dtype=args.float_type).to(args.device)
        test_y_tensor = torch.tensor(test_y, dtype=args.float_type).to(args.device)
        val_x_tensor = torch.tensor(val_x, dtype=args.float_type).to(args.device)
        val_y_tensor = torch.tensor(val_y, dtype=args.float_type).to(args.device)

    elif type == 'ori':
        train_x_tensor = torch.tensor(train_x[:, :, [6]], dtype=torch.float32).to(args.device)
        train_y_tensor = torch.tensor(train_y[:, :, [4]], dtype=torch.float32).to(args.device)
        test_x_tensor = torch.tensor(test_x[:, :, [6]], dtype=torch.float32).to(args.device)
        test_y_tensor = torch.tensor(test_y[:, :, [4]], dtype=torch.float32).to(args.device)
        val_x_tensor = torch.tensor(val_x[:, :, [6]], dtype=torch.float32).to(args.device)
        val_y_tensor = torch.tensor(val_y[:, :, [4]], dtype=torch.float32).to(args.device)

    elif type == 'std':  # x had been standardized, y had not been standardized before
        assert Scale is not None
        train_x_np = train_x[:, :, [5]]
        train_y_np = train_y[:, :, [4]]
        test_x_np = test_x[:, :, [5]]
        test_y_np = test_y[:, :, [4]]
        val_x_np = val_x[:, :, [5]]
        val_y_np = val_y[:, :, [4]]

        train_y_norm = Scale.transform(train_y_np)
        test_y_norm = Scale.transform(test_y_np)
        val_y_norm = Scale.transform(val_y_np)

        # train_x_denorm = Scale.inverse_transform(train_x_np)
        # test_x_denorm = Scale.inverse_transform(test_x_np)
        # val_x_denorm = Scale.inverse_transform(val_x_np)

        train_x_tensor = torch.tensor(train_x_np, dtype=args.float_type).to(args.device)
        train_y_tensor = torch.tensor(train_y_norm, dtype=args.float_type).to(args.device)
        test_x_tensor = torch.tensor(test_x_np, dtype=args.float_type).to(args.device)
        test_y_tensor = torch.tensor(test_y_norm, dtype=args.float_type).to(args.device)
        val_x_tensor = torch.tensor(val_x_np, dtype=args.float_type).to(args.device)
        val_y_tensor = torch.tensor(val_y_norm, dtype=args.float_type).to(args.device)


    return train_x_tensor, train_y_tensor, test_x_tensor, test_y_tensor, val_x_tensor, val_y_tensor


def get_statistical(file_path):

    statistics_data = torch.load(os.path.join(file_path, "mean_std_mini.pt"))

    train_diff_mean = statistics_data['diff_mean']
    train_diff_std = statistics_data['diff_std']
    train_min = statistics_data['mini']
    train_mean = statistics_data['stdn_mean']
    train_std = statistics_data['stdn_std']

    return train_diff_mean, train_diff_std, train_min, train_mean, train_std


def prepare_cwt_tensors(args):
    """
    Load and organize all .npz files by status (train/val/test),
    normalization type (std/ori), and 'k' group (k1, k12, etc.).

    Returns:
        data_dict: dict {
            'train'/'val'/'test': {
                'std'/'ori': {
                    'k1'/'k12'/...: {
                        'power': torch.Tensor, shape (N_total, C, S, T),
                        'freqs': np.ndarray, shape (S,)
                    }
                }
            }
        }
    """

    # regex pattern: b000_k12_std.npz
    pattern = re.compile(r"b(\d+)_k(\d+)_(std|ori|logstd)\.npz")
    data_dict = {}

    for status in ['train', 'val', 'test']:
        folder_path = os.path.join(args.cwt_dir, args.data_set, status)
        if not os.path.exists(folder_path):
            print(f"Warning: folder not found: {folder_path}")
            continue

        data_dict[status] = {}

        files = [f for f in os.listdir(folder_path) if f.endswith(".npz") and pattern.match(f)]
        grouped = {}

        for f in files:
            match = pattern.match(f)
            if not match:
                continue

            batch_num = int(match.group(1))
            k_val = f'k{match.group(2)}'
            norm_type = match.group(3)  # 'std' or 'ori'

            key = (norm_type, k_val)
            if key not in grouped:
                grouped[key] = []

            grouped[key].append((batch_num, f))

        for (norm_type, k_val), file_list in grouped.items():
            sorted_files = sorted(file_list, key=lambda x: x[0])

            all_power = []
            freqs_ref = None

            for _, fname in sorted_files:
                fpath = os.path.join(folder_path, fname)
                data = np.load(fpath)
                power = torch.tensor(data["power"], dtype=torch.float32)
                all_power.append(power)

                if freqs_ref is None:
                    freqs_ref = data["freqs"]

            power_tensor = torch.cat(all_power, dim=0).to(args.device)

            if norm_type not in data_dict[status]:
                data_dict[status][norm_type] = {}

            data_dict[status][norm_type][k_val] = {
                'power': power_tensor,
                'freqs': freqs_ref
            }

    return data_dict



def get_cwt_data(args):
    """
    Generate cwt_{split}_x and feq_{split}_x for each data split from data_dict,
    merging K into one dimension.
    """

    data_dict = prepare_cwt_tensors(args)
    norm_type = args.norm_type

    cwt_power_train_x, cwt_power_val_x, cwt_power_test_x, cwt_feq_train_x, cwt_feq_val_x, cwt_feq_test_x = None, None, None, None, None, None

    for split in ['train', 'val', 'test']:
        if split not in data_dict or norm_type not in data_dict[split]:
            print(f"[Warning] Missing {split}/{norm_type} data. Skipped.")
            continue

        k_dict = data_dict[split][norm_type]
        power_list = []
        freqs_list = []

        for k_name in sorted(k_dict.keys()):
            entry = k_dict[k_name]
            power = entry['power']  # shape: (B, C, S, T)
            freqs = entry['freqs']  # shape: (S,)

            # Add K dimension
            power = power.unsqueeze(-1)           # -> (B, C, S, T, 1)
            freqs = freqs[:, np.newaxis]          # -> (S, 1)

            power_list.append(power)
            freqs_list.append(freqs)

        # Merge K dimension
        power_tensor = torch.cat(power_list, dim=-1)   # (B, C, S, T, K)
        freq_tensor = np.concatenate(freqs_list, axis=1)  # (S, K)

        if split == 'train':
            cwt_power_train_x, cwt_feq_train_x = power_tensor, freq_tensor
        elif split == 'val':
            cwt_power_val_x, cwt_feq_val_x = power_tensor, freq_tensor
        elif split == 'test':
            cwt_power_test_x, cwt_feq_test_x = power_tensor, freq_tensor

    return cwt_power_train_x, cwt_power_val_x, cwt_power_test_x, cwt_feq_train_x, cwt_feq_val_x, cwt_feq_test_x


def get_time_feature(args, file_path):
    """
    Generate time features for train, test, and validation sets.

    Returns:
        train_dec, test_dec, val_dec: Tensors containing time features.
    """

    train_y = np.load(os.path.join(file_path, "train_y.npy"))
    test_y = np.load(os.path.join(file_path, "test_y.npy"))
    val_y = np.load(os.path.join(file_path, "val_y.npy"))

    # Extract time features (cos and sin of date)
    train_time_y = train_y[:, :, [1, 2]]
    test_time_y = test_y[:, :, [1, 2]]
    val_time_y = val_y[:, :, [1, 2]]

    return torch.tensor(train_time_y, dtype=args.float_type).to(args.device), \
           torch.tensor(test_time_y, dtype=args.float_type).to(args.device), \
           torch.tensor(val_time_y, dtype=args.float_type).to(args.device)


def smooth_conv(seq, kernel_size, mode='replicate'):
    """
    Apply 1D smoothing convolution to input of shape (B, T, C)
    """
    if not isinstance(seq, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if seq.dim() != 3:
        raise ValueError("Input must have shape (B, T, C)")

    B, T, C = seq.shape

    # Reshape to (B * C, 1, T)
    seq_reshaped = seq.permute(0, 2, 1).reshape(B * C, 1, T)

    # Create kernel: shape (1, 1, kernel_size)
    kernel = torch.ones(1, 1, kernel_size, dtype=seq.dtype, device=seq.device) / kernel_size

    # Pad: symmetric padding
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    seq_padded = F.pad(seq_reshaped, (pad_left, pad_right), mode=mode)  # seq_padded shape: (B * C, 1, T + kernel_size - 1)

    # Convolve
    out = F.conv1d(seq_padded, kernel)  # out shape: (B * C, 1, T)

    # Reshape back to (B, T, C)
    out = out.view(B, C, T).permute(0, 2, 1)    # out shape: (B, T, C)
    return out

def generate_cwt_from_tensor(
        x_tensor,  # torch.Tensor, shape (B, T, C)
        kernel_list,  # list[int], e.g., [1, 24, 48]
        fs,  # float, sampling rate
        num_scales,  # int, number of scales for CWT
        save_dir,  # str, directory to save npz
        data_set,  # str
        status,  # str, 'train', 'val', etc.
        wavelet_name='cgau7',
        norm_type="std",  # str
        device=torch.device("cpu")
):
    """
    Perform CWT on multi-resolution kernels and save to .npz

    Result shape: (B, C, S, T, N), where N = len(kernel_list)
    """
    B, T, C = x_tensor.shape

    results = []
    freq_list = []
    freqs_ref = None

    for k in kernel_list:
        print(k)
        if k == 1:
            smooth_seq = x_tensor  # (B, T, C)
        else:
            smooth_seq = smooth_conv(x_tensor, k, 'replicate')  # (B, T, C)

        # First-order diff
        diff_seq = torch.diff(smooth_seq, dim=1, prepend=smooth_seq[:, :1])  # (B, T, C)

        # → numpy → (B, C, T)
        x_np = diff_seq.detach().cpu().numpy().astype(np.float32)
        x_np = np.transpose(x_np, (0, 2, 1))  # (B, C, T)
        x_reshaped = x_np.reshape(B * C, T)

        # CWT
        scales = np.logspace(1, np.log10(T // 2), num=num_scales, base=10)
        coeff, freqs = pywt.cwt(x_reshaped, scales, wavelet_name, sampling_period=1/fs)
        power = np.abs(coeff) ** 2  # (S, B*C, T)
        power = power.transpose(1, 0, 2).reshape(B, C, len(scales), T)  # (B, C, S, T)
        results.append(power[..., np.newaxis])  # (B, C, S, T, 1)

        freq_list.append(freqs[..., np.newaxis])

    # → (B, C, S, T, N)
    power_all = np.concatenate(results, axis=-1)
    freqs_all = np.concatenate(freq_list, axis=-1)

    power_all_tensor = torch.tensor(power_all, dtype=torch.float32).to(device)
    freqs_all_tensor = torch.tensor(freqs_all, dtype=torch.float32).to(device)

    # Save
    save_path = os.path.join(save_dir, data_set, status)
    os.makedirs(save_path, exist_ok=True)
    kernel_name = "_".join(map(str, kernel_list))
    filename = os.path.join(save_path, f"k{kernel_name}_{norm_type}_scale{num_scales}_fs{fs}.npz")
    np.savez_compressed(filename, power=power_all, freqs=freqs_all)
    print(f"[Saved] {filename} | shape={power_all.shape}, freqs={freqs_all.shape}")

    return power_all_tensor, freqs_all_tensor



def get_cwt_tensor(train_x, val_x, args, test_x=None):
    """
    Generate or load cwt_{split}_x and feq_{split}_x for each data split.
    """

    def load_or_generate(x_tensor, status):
        kernel_list = [1] + args.resolution_size
        kernel_name = "_".join(map(str, kernel_list))
        filename = os.path.join(
            args.cwt_dir,
            args.data_set,
            status,
            f"k{kernel_name}_{args.norm_type}_scale{args.num_scales}_fs{args.fs}.npz"
        )

        if os.path.exists(filename):
            data = np.load(filename)
            power = torch.tensor(data["power"], dtype=args.float_type).to(x_tensor.device)
            freqs = data["freqs"]
        else:
            power, freqs = generate_cwt_from_tensor(
                x_tensor=x_tensor,
                fs=args.fs,
                num_scales=args.num_scales,
                save_dir=args.cwt_dir,
                data_set=args.data_set,
                status=status,
                kernel_list=kernel_list,
                norm_type=args.norm_type,
                device=x_tensor.device,
            )

        return power, freqs

    if test_x is not None and args.status == "train":

        cwt_test_x, feq_test_x = load_or_generate(test_x, status="test")
        cwt_train_x, feq_train_x = load_or_generate(train_x, status="train")
        cwt_val_x, feq_val_x = load_or_generate(val_x, status="val")

        feq_test_x = torch.tensor(feq_test_x, dtype=args.float_type).unsqueeze(0).repeat(cwt_test_x.shape[0], 1, 1)
        feq_train_x = torch.tensor(feq_train_x, dtype=args.float_type).unsqueeze(0).repeat(cwt_train_x.shape[0], 1, 1)
        feq_val_x = torch.tensor(feq_val_x, dtype=args.float_type).unsqueeze(0).repeat(cwt_val_x.shape[0], 1, 1)

        return cwt_train_x, cwt_val_x, cwt_test_x, feq_train_x, feq_val_x, feq_test_x

    elif args.status == 'test':
        if test_x is None:
            raise ValueError("test_x must be provided when status is 'test'")

        cwt_test_x, feq_test_x = load_or_generate(test_x, status="test")
        feq_test_x = torch.tensor(feq_test_x, dtype=args.float_type).unsqueeze(0).repeat(cwt_test_x.shape[0], 1, 1)

        return cwt_test_x, feq_test_x

    # TRAIN
    cwt_train_x, feq_train_x = load_or_generate(train_x, status="train")
    cwt_val_x, feq_val_x = load_or_generate(val_x, status="val")

    feq_train_x = torch.tensor(feq_train_x, dtype=args.float_type).unsqueeze(0).repeat(cwt_train_x.shape[0], 1, 1)
    feq_val_x = torch.tensor(feq_val_x, dtype=args.float_type).unsqueeze(0).repeat(cwt_val_x.shape[0], 1, 1)

    return cwt_train_x, cwt_val_x, feq_train_x, feq_val_x


def get_data_macnn(args, file_path, type='all', Scale=None):
    """Load and preprocess data for training and testing."""

    test_x = np.load(os.path.join(file_path, "test_x.npy"))
    test_y = np.load(os.path.join(file_path, "test_y.npy"))
    train_x = np.load(os.path.join(file_path, "train_x.npy"))
    train_y = np.load(os.path.join(file_path, "train_y.npy"))
    val_x = np.load(os.path.join(file_path, "val_x.npy"))
    val_y = np.load(os.path.join(file_path, "val_y.npy"))

    assert type in ['diff']
    enc_selected_indices = [0, 1, 2, 3, 4, 13, 14, 15]
    dec_selected_indices = [1, 2]

    if type == 'diff':
        train_x_tensor = torch.tensor(train_x[:, :, enc_selected_indices], dtype=args.float_type).to(args.device)
        test_x_tensor = torch.tensor(test_x[:, :, enc_selected_indices], dtype=args.float_type).to(args.device)
        val_x_tensor = torch.tensor(val_x[:, :, enc_selected_indices], dtype=args.float_type).to(args.device)

        train_x_dec_tensor = torch.tensor(train_y[:, :, dec_selected_indices], dtype=args.float_type).to(args.device)
        test_x_dec_tensor = torch.tensor(test_y[:, :, dec_selected_indices], dtype=args.float_type).to(args.device)
        val_x_dec_tensor = torch.tensor(val_y[:, :, dec_selected_indices], dtype=args.float_type).to(args.device)

        train_y_np = train_y[:, :, [0]]
        test_y_np = test_y[:, :, [0]]
        val_y_np = val_y[:, :, [0]]

        pre_train_y_np = train_y[:, :, [3]][:, [0], 0]
        pre_test_y_np = test_y[:, :, [3]][:, [0], 0]
        pre_val_y_np = val_y[:, :, [3]][:, [0], 0]

        Scale._set_diff_params(pre_train_y_np, type='train')
        Scale._set_diff_params(pre_test_y_np, type='test')
        Scale._set_diff_params(pre_val_y_np, type='val')

        Scale._set_target(train_y[:, :, [4]], type='train')
        Scale._set_target(test_y[:, :, [4]], type='test')
        Scale._set_target(val_y[:, :, [4]], type='val')

        train_y_tensor = torch.tensor(train_y_np, dtype=args.float_type).to(args.device)
        test_y_tensor = torch.tensor(test_y_np, dtype=args.float_type).to(args.device)
        val_y_tensor = torch.tensor(val_y_np, dtype=args.float_type).to(args.device)

        return (train_x_tensor, train_x_dec_tensor, train_y_tensor,
                test_x_tensor, test_x_dec_tensor, test_y_tensor,
                val_x_tensor, val_x_dec_tensor, val_y_tensor)
