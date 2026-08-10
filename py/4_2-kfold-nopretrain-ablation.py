# No-Pretrain Ablation: identical recipe to the production k-fold fine-tuning
# script (py/4_2-kfold-finetune-temporal-unet.py), EXCEPT the pretrained-weight
# loading step is skipped (model starts from random init instead of the
# SPoRT-LIS-pretrained checkpoint). Everything else -- architecture, encoder
# unfreezing, LR schedule, loss, 30 epochs -- is identical, so the ablation
# isolates the effect of the pretrained starting point under the exact same
# fixed compute budget used operationally.
#
# Writes to results-kfold-{depth}-nopretrain-ablation/ -- NEVER touches the
# production results-kfold-{depth}/ directories.
#
# Run:  DEPTH_FLAG=middle  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python py/4_2-kfold-nopretrain-ablation.py
#       DEPTH_FLAG=shallow CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python py/4_2-kfold-nopretrain-ablation.py

import sys
import time
import datetime
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import re
import warnings
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *
from define_hyperparams import *

DEPTH = os.environ.get("DEPTH_FLAG", "middle")
# LR_SCHEDULE: "twophase" (default, matches production recipe exactly: 1e-5 for
# 5 epochs then 1e-7) or "heldlr" (diagnostic only -- LR held at 1e-5 for all 30
# epochs, to see how much more a from-scratch model learns when not throttled
# by a schedule tuned for fine-tuning near a good solution). NOT the official
# ablation -- writes to a separate -heldlr output dir, single-fold spot-check.
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "twophase")
FOLDS = [int(x) for x in os.environ.get("FOLDS", "").split(",")] if os.environ.get("FOLDS") else None
assert DEPTH in ("shallow", "middle")
BASE = "/data/ssd2/soil-moisture-ml"

num_epochs = 30
criterion = L1WithWindowedDerivativeAndBiasLoss(lambda_deriv=1, lambda_bias=0.5)

def compute_kge(yhat, yobs):
    # inline KGE (Gupta et al. 2009) -- hydroeval is not installed in this env
    m = ~np.isnan(yhat) & ~np.isnan(yobs)
    if m.sum() < 2:
        return np.nan
    s, o = yhat[m], yobs[m]
    if s.std() == 0 or o.std() == 0 or o.mean() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)

def get_first_file_path(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return os.path.join(directory, files[0]) if files else None

def run_fold(fold, train_path, val_path):
    print(f"\n========== [{DEPTH}] NO-PRETRAIN ABLATION FOLD {fold} ==========")
    model_name = f'uNET_nopretrain_{DEPTH}_fold_{fold}_{datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S_%f")}'
    start_time = time.time()

    template = StreamflowTrainDatasetTemplate(
        csv_file=get_first_file_path(f'{BASE}/full-dataloader-{DEPTH}')
    )
    input_size = template.x_tensor.size()[2]

    train_data = StreamflowTrainDatasetXY(
        path_csv=train_path,
        dim_1=template.x_tensor.shape[1],
        dim_2=template.x_tensor.shape[2]
    )
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=64)

    test_data = StreamflowTrainDatasetXY_Test(
        path_csv=val_path,
        dim_2=template.x_tensor.shape[2]
    )
    data_loader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    model = uNet(num_classes, input_size, max_channels, dropout).to('cuda')
    # NO pretrained-weight loading -- model starts from random init (uNet's own
    # default initialization). This is the ONLY difference from the production
    # fine-tuning script. All parameters are trainable by default (no freezing
    # needed since there's no pretrained encoder to protect).
    print("✗ Skipping pretrained-weight loading (random init) -- ablation condition")

    tag = "" if LR_SCHEDULE == "twophase" else f"-{LR_SCHEDULE}"
    out_dir = f'{BASE}/results-kfold-{DEPTH}-nopretrain-ablation{tag}/fold_{fold}_{model_name}'

    initial_fast_lr = 1e-5
    slow_lr = 1e-7 if LR_SCHEDULE == "twophase" else initial_fast_lr   # "heldlr": never switch, stays at 1e-5
    lr_switch_epoch = 5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_fast_lr)

    losses = []
    test_kge_df = pd.DataFrame(columns=['epoch', 'median_kge', 'lower_kge', 'upper_kge'])
    test_losses_df = pd.DataFrame(columns=['epoch', 'loss'])
    kge_out = []
    path_out = []

    model.eval()
    test_kges = []
    for X, y, path in data_loader_test:
        with torch.no_grad():
            outputs = model(X.to('cuda'))
            yhat = outputs.cpu().view(-1).numpy()
            yobs = y.cpu().view(-1).numpy()
            kge = compute_kge(yhat, yobs)
        test_kges.append(kge)

    test_kge_df = pd.concat([test_kge_df, pd.DataFrame([[0, np.nanmedian(test_kges), np.nanpercentile(test_kges, 25), np.nanpercentile(test_kges, 75)]], columns=['epoch', 'median_kge', 'lower_kge', 'upper_kge'])], ignore_index=True)
    print(f"✔ Initial (random-init) Validation KGE: {np.nanmedian(test_kges)}")

    for epoch in range(num_epochs):
        if epoch == lr_switch_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = slow_lr
            print(f"✔ Switched to slow LR: {slow_lr}")

        model.train()
        batchloss = []
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to('cuda'))
            loss = criterion(outputs, y_batch.to('cuda'))
            loss.backward()
            optimizer.step()
            batchloss.append(loss.item())

        losses.append(np.mean(batchloss))
        print(f"Epoch: {epoch}, Loss: {np.mean(batchloss):.5f}")

        if epoch % 5 == 0 and epoch != 0:
            model.eval()
            test_kges = []
            test_loss_batch = []
            for X, y, path in data_loader_test:
                with torch.no_grad():
                    outputs = model(X.to('cuda'))
                yhat = outputs.cpu().view(-1).numpy()
                yobs = y.cpu().view(-1).numpy()
                kge = compute_kge(yhat, yobs)
                test_kges.append(kge)
                loss = criterion(outputs, y.to('cuda'))
                test_loss_batch.append(loss.item())

            test_kge_df = pd.concat([test_kge_df, pd.DataFrame(
                [[epoch, np.nanmedian(test_kges), np.nanpercentile(test_kges, 25), np.nanpercentile(test_kges, 75)]],
                columns=['epoch', 'median_kge', 'lower_kge', 'upper_kge'])], ignore_index=True)
            test_losses_df = pd.concat([
                test_losses_df,
                pd.DataFrame([[epoch, np.mean(test_loss_batch)]], columns=['epoch', 'loss'])
            ], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{out_dir}/model.pt')
    pd.DataFrame(losses, columns=['train_loss']).to_csv(f'{out_dir}/train_losses.csv', index=False)
    test_kge_df.to_csv(f'{out_dir}/test_kge.csv', index=False)

    for i, (X, y, path) in enumerate(data_loader_test):
        with torch.no_grad():
            outputs = model(X.to('cuda'))
            yhat = outputs.cpu().view(-1).numpy()
            yobs = y.view(-1).numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                kge = compute_kge(yhat, yobs)
            kge_out.append(float(kge))
            path_out.append(path[0])

    runtime = time.time() - start_time
    print(f'⏱ [{DEPTH}] fold {fold} runtime: {runtime / 60:.2f} minutes')

    pd.DataFrame({'path': path_out, 'kge': kge_out}).to_csv(f'{out_dir}/ungaged_kge_results.csv', index=False)

    n_param = sum(p.numel() for p in model.parameters())
    pd.DataFrame({
        'var': ['model_name', 'depth', 'max_channels', 'input_size', 'hidden_size', 'batch_size', 'num_layers',
                'num_classes', 'linear_nodes', 'initial_learning_rate', 'dropout', 'num_epochs', 'runtime_secs',
                'median_kge', 'nparam', 'pretrained'],
        'vals': [model_name, DEPTH, max_channels, input_size, hidden_size, batch_size, num_layers, num_classes,
                 linear_nodes, initial_fast_lr, dropout, num_epochs, runtime, np.nanmedian(kge_out), n_param, False]
    }).to_csv(f'{out_dir}/model_hyperparams.csv', index=False)

    del model
    torch.cuda.empty_cache()

split_dir = f'{BASE}/split-definitions-kfold-{DEPTH}'
train_paths = sorted(glob.glob(f"{split_dir}/train_split_fold_*.csv"))
val_paths = sorted(glob.glob(f"{split_dir}/validation_split_fold_*.csv"))
assert len(train_paths) == len(val_paths), "Mismatch in train/val split counts"
assert len(train_paths) == 10, f"Expected 10 folds, found {len(train_paths)}"

for fold, (train_path, val_path) in enumerate(zip(train_paths, val_paths), start=1):
    if FOLDS is not None and fold not in FOLDS:
        continue
    run_fold(fold, train_path, val_path)
