# K-Fold Cross-Validation Script for Soil Moisture Modeling

import sys
import time
import datetime
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
import re
import hydroeval as he
import warnings
import glob

# Append custom module path (this script's own dir, where model_config lives)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *
from define_hyperparams import *

num_epochs = 30
criterion = L1WithWindowedDerivativeAndBiasLoss(lambda_deriv=1, lambda_bias=0.5)

def compute_kge(yhat, yobs):
    mask = ~np.isnan(yhat) & ~np.isnan(yobs)
    if np.sum(mask) < 2:
        return np.nan
    return he.kge(yhat[mask].reshape(-1, 1), yobs[mask].reshape(-1, 1))[0]

def unfreeze_encoder_layer(model, layer_name):
    for name, param in model.named_parameters():
        if name.startswith(layer_name):
            param.requires_grad = True
    print(f"\u2714 Unfroze layer: {layer_name}")

def get_first_file_path(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return os.path.join(directory, files[0]) if files else None

def run_fold(fold, train_path, val_path):
    print(f"\n========== FOLD {fold} ==========")
    model_name = f'uNET_fold_{fold}_{datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S_%f")}'
    start_time = time.time()

    template = StreamflowTrainDatasetTemplate(
        csv_file=get_first_file_path('/data/ssd2/soil-moisture-ml/full-dataloader-middle')
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
    pretrained_path = "/data/ssd2/soil-moisture-ml/results-pretrain/uNET_middle_18339_sites_08_09_2025-23_18_00_709581/model.pt"
    model.load_state_dict(torch.load(pretrained_path, map_location='cuda'))
    print(f"\u2714 Loaded pre-trained model from {pretrained_path}")

    out_dir = f'/data/ssd2/soil-moisture-ml/results-kfold-middle/fold_{fold}_{model_name}'

    encoder_layers = ["encoder1", "encoder2", "encoder3", "encoder4"]
    for name, param in model.named_parameters():
        if any(name.startswith(enc) for enc in encoder_layers):
            param.requires_grad = True

    initial_fast_lr = 1e-5
    slow_lr = 1e-7
    lr_switch_epoch = 5
    warmup_epochs = 0
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_fast_lr)

    unfreeze_schedule = {200: ["encoder4"]}

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
    print(f"\u2714 Initial Validation KGE: {np.nanmedian(test_kges)}")

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr = initial_fast_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == lr_switch_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = slow_lr
            print(f"\u2714 Switched to slow LR: {slow_lr}")

        if epoch in unfreeze_schedule:
            for layer_name in unfreeze_schedule[epoch]:
                unfreeze_encoder_layer(model, layer_name)

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

            plt.figure(figsize=(10, 6))
            for y_line in np.arange(0.1, 1.0, 0.1):
                plt.axhline(y=y_line, linestyle='--', color='gray', linewidth=0.5)
            plt.plot(test_kge_df['epoch'], test_kge_df['median_kge'], label='Median KGE', color='green', linewidth=2)
            plt.plot(test_kge_df['epoch'], test_kge_df['lower_kge'], label='25th Percentile', color='red', linewidth=1.5)
            plt.plot(test_kge_df['epoch'], test_kge_df['upper_kge'], label='75th Percentile', color='blue', linewidth=1.5)
            plt.xlabel('Epoch')
            plt.ylabel('Kling-Gupta Efficiency (KGE)')
            plt.title('KGE Performance Over Epochs')
            plt.grid(True, linestyle=':')
            plt.legend()
            plt.ylim(0, 0.8)
            plt.tight_layout()
            #plt.savefig(f'{out_dir}/kge_plot_epoch_{epoch}.png')
            plt.show()

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/plots/', exist_ok=True)
    os.makedirs(f'{out_dir}/hydrograph_data/', exist_ok=True)
    torch.save(model.state_dict(), f'{out_dir}/model.pt')
    pd.DataFrame(losses, columns=['train_loss']).to_csv(f'{out_dir}/train_losses.csv', index=False)
    test_kge_df.to_csv(f'{out_dir}/test_kge.csv', index=False)

    for i, (X, y, path) in enumerate(data_loader_test):
        with torch.no_grad():
            outputs = model(X.to('cuda'))
            yhat = outputs.cpu().view(-1).numpy()
            yobs = y.view(-1).numpy()
            site_id = re.search(r'basin-group-(.*?)-', path[0]).group(1)

            plotting_data = pd.DataFrame({'yhat': yhat, 'yobs': yobs})
            sns.lineplot(data=plotting_data).set_title(f'Site: {site_id}')
            plt.xlabel("Days")
            plt.ylabel("Soil Moisture")
            plt.savefig(f'{out_dir}/plots/plot_{site_id}.png')
            plt.clf()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                kge = he.kge(yhat.reshape(-1, 1), yobs.reshape(-1, 1))

            kge_out.append(kge[0].item())
            path_out.append(path[0])
            plotting_data['kge'] = kge[0].item()
            plotting_data['path'] = path[0]
            plotting_data.to_csv(f'{out_dir}/hydrograph_data/data_{i}.csv')

    hist, bin_edges = np.histogram(kge_out, bins=np.linspace(0, 1, 11))
    plt.bar(bin_edges[:-1], hist, width=0.1)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.savefig(f'{out_dir}/kge_hist.png')
    plt.clf()

    kge_sorted = np.sort(np.array(kge_out))
    cdf = np.arange(1, len(kge_sorted) + 1) / len(kge_sorted)
    median_kge = np.nanmedian(kge_sorted)
    median_cdf_value = np.interp(median_kge, kge_sorted, cdf)

    plt.figure(figsize=(8, 6))
    plt.plot(kge_sorted, cdf, label="CDF")
    plt.axvline(x=median_kge, color='r', linestyle='--', label=f'Median: {median_kge:.2f}')
    plt.scatter(median_kge, median_cdf_value, color='red')
    plt.xlabel("KGE")
    plt.ylabel("CDF")
    plt.title("Cumulative Distribution Function (CDF) - KGE")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.savefig(f'{out_dir}/kge_cdf.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, label='Train Loss', color='blue')
    plt.plot(test_losses_df['epoch'], test_losses_df['loss'], label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Test Loss Over Epochs')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/loss_comparison.png')
    plt.clf()

    runtime = time.time() - start_time
    print(f'\u23F1 Runtime: {runtime / 60:.2f} minutes, or {runtime / 3600:.2f} hours')

    pd.DataFrame({'path': path_out, 'kge': kge_out}).to_csv(f'{out_dir}/ungaged_kge_results.csv', index=False)

    n_param = sum(p.numel() for p in model.parameters())
    pd.DataFrame({
        'var': ['model_name', 'max_channels', 'input_size', 'hidden_size', 'batch_size', 'num_layers', 'num_classes', 'linear_nodes', 'initial_learning_rate', 'dropout', 'num_epochs', 'runtime_secs', 'median_kge', 'nparam'],
        'vals': [model_name, max_channels, input_size, hidden_size, batch_size, num_layers, num_classes, linear_nodes, initial_fast_lr, dropout, num_epochs, runtime, median_kge, n_param]
    }).to_csv(f'{out_dir}/model_hyperparams.csv', index=False)

    del model
    torch.cuda.empty_cache()

# Run all folds
split_dir = '/data/ssd2/soil-moisture-ml/split-definitions-kfold-middle'
train_paths = sorted(glob.glob(f"{split_dir}/train_split_fold_*.csv"))
val_paths = sorted(glob.glob(f"{split_dir}/validation_split_fold_*.csv"))

assert len(train_paths) == len(val_paths), "Mismatch in train/val split counts"

for fold, (train_path, val_path) in enumerate(zip(train_paths, val_paths), start=1):
    run_fold(fold, train_path, val_path)