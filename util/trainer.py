#!/usr/bin/env python
# encoding: utf-8

import copy
import time
import os
import pickle

import torch
from torch import nn, optim
from torch.utils import data
from torch.autograd import grad

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Union

from tqdm import tqdm

from util.scale import Scale
from util.evaluation import metric_rolling
from util.tools import to_string, adjust_learning_rate, model_predictor, model_predictor_test

def Trainer(model: nn.Module, train_data, validation_data=None,
            criterion: nn.Module = nn.MSELoss(), early_stop_threshold = 4,
            scaler: Scale = None, shuffle: bool = False, verbose: bool = False,
            checkpoint: int = None, save_model: bool = False,
            args = None):

    print('Data: {}, Model: {}, epochs: {}, batch_size: {}, init_lr: {}, early_stop: {}, criterion: {}, norm_type: {}, align_loss_ratio: {}, diversity_loss_ratio: {}'.format(
        args.data_set, type(model).__name__, args.epochs, args.batchsize, args.learning_rate,
        early_stop_threshold, type(criterion).__name__, args.norm_type, args.align_loss_ratio, args.diversity_loss_ratio))

    train_dataset = data.TensorDataset(*train_data)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=shuffle)

    best_val_loss = float('inf')
    best_model_state = None

    if early_stop_threshold < 0:
        early_stop_threshold = args.epochs

    early_stop = 0

    filtered_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(filtered_parameters, lr=args.learning_rate)

    for epoch in range(args.epochs):

        print_loss_total = 0

        model.train()
        start_time = time.time()
        args.status = 'train'

        for step, batch_train in enumerate(train_loader):
            batch_x, batch_y = batch_train[:-1], batch_train[-1]

            args.batchstep = step
            if hasattr(model, 'args'):
                model.args = args

            batch_y_hat, align_loss, diversity_loss, _, _ = model(*batch_x)
            loss = criterion(batch_y_hat, batch_y) + args.align_loss_ratio * align_loss + args.diversity_loss_ratio * diversity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss_total += loss.item()

        if checkpoint is not None and checkpoint > 0 and epoch % checkpoint == 0:
            filename = './data/' + '{}.{}.checkpoint'.format(type(model).__name__, epoch)
            torch.save({'ml': model.state_dict()}, filename)

        if not verbose:
            continue

        model.eval()
        end_time = time.time()
        time_cost = end_time - start_time
        epoch_results = [epoch, str(f"{time_cost:.2f}") + 's', optimizer.param_groups[0]['lr']]

        if validation_data is not None:

            args.status = 'val'

            y_val = validation_data[-1].detach()
            y_val_hat = model_predictor(model, validation_data[:-1], args)

            if scaler is not None:
                y_val = scaler.inverse_transform(y_val, args.norm_type, 'val', 'real')
                y_val_hat = scaler.inverse_transform(y_val_hat, args.norm_type, 'val', 'predict')

            y_val_hat = (y_val_hat + torch.abs(y_val_hat)) / 2  # avoid negative values

            val_loss = criterion(y_val_hat, y_val)
            normal_val_results = metric_rolling(f'out{args.output_len}_ro{args.roll}', y_val_hat, y_val, args.output_len, args.output_len)  # Evaluation
            roll_train_results = metric_rolling(f'out{args.output_len}_ro{args.roll}', y_val_hat, y_val, args.roll, args.output_len)  # Evaluation

            val_flattened_results = [item for value in normal_val_results.values() for item in
                                     (value if isinstance(value, list) else [value])]

            roll_flattened_results = [item for value in roll_train_results.values() for item in
                                      (value if isinstance(value, list) else [value])]

            epoch_results += ['|', val_loss.item()] + val_flattened_results + ['|'] + roll_flattened_results

            if save_model and (val_loss < best_val_loss) and epoch >= 20:
                # print(f"Validation loss improved from {best_val_loss} to {val_loss}")
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())

        # early stop
        if val_loss > best_val_loss and epoch >= 20:
            early_stop += 1

        else:
            early_stop = 0

        print(to_string(*epoch_results))

        if early_stop >= early_stop_threshold:
            break

        adjust_learning_rate(optimizer, epoch + 1, args)


    if save_model and best_model_state is not None:
        model.load_state_dict(best_model_state)

        if args.mode == 'train':
            op_model_path = f"./save_model/{args.model_name}/{args.data_set}/pl{args.output_len}_in{args.input_len}_ro{args.roll}_ns{args.norm_type}.pkl"
            if not os.path.exists(os.path.dirname(op_model_path)):
                os.makedirs(os.path.dirname(op_model_path))

            current_val_loss = best_val_loss
            save_new_model = True
            previous_val_loss = float('inf')

            if os.path.exists(op_model_path):
                with open(op_model_path, "rb") as f:
                    saved_data = pickle.load(f)
                    previous_val_loss = saved_data.get("val_loss", float("inf"))

                if previous_val_loss <= current_val_loss:
                    print(f"Previous model performs better (val loss: {previous_val_loss:.4f}) "
                          f"than current model (val loss: {current_val_loss:.4f}). Not saving.")
                    save_new_model = False
                else:
                    print(f"Current model performs better (val loss: {current_val_loss:.4f}) "
                          f"than previous model (val loss: {previous_val_loss:.4f}). Saving new model.")

            if save_new_model:
                with open(op_model_path, "wb") as f:
                    pickle.dump({
                        "model_state_dict": best_model_state,
                        "val_loss": current_val_loss
                    }, f)
                print(f"Model saved to {op_model_path} with val loss {current_val_loss:.4f}")

    return model
