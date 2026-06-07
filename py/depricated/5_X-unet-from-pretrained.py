# Full Fine-Tuning Script with Two-Phase LR Schedule + Gradual Unfreezing + Warmup + Full Reporting

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

# Append custom module path
sys.path.append('/home/zhoylman/soil-moisture-ml/py')
from model_config import *
from model_config import *
from define_hyperparams import *

num_epochs = 30
criterion = L1WithWindowedDerivativeAndBiasLoss(lambda_deriv=1, lambda_bias=0.5)

def compute_kge(yhat, yobs):
    """Safe KGE wrapper for hydroeval.kge() with NaN handling."""
    mask = ~np.isnan(yhat) & ~np.isnan(yobs)
    if np.sum(mask) < 2:
        return np.nan
    return he.kge(yhat[mask].reshape(-1, 1), yobs[mask].reshape(-1, 1))[0]

# Gradually unfreeze encoder layers
def unfreeze_encoder_layer(model, layer_name):
    for name, param in model.named_parameters():
        if name.startswith(layer_name):
            param.requires_grad = True
    print(f"\u2714 Unfroze layer: {layer_name}")

# Fine-tuning procedure
def train_and_evaluate(site_number):
    print(f"Training and evaluating model for site_number: {site_number}")
    model_name = f'uNET_{site_number}_sites_{datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S_%f")}'
    start_time = time.time()

    def get_first_file_path(directory):
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return os.path.join(directory, files[0]) if files else None

    template = StreamflowTrainDatasetTemplate(
        csv_file=get_first_file_path('/data/ssd2/soil-moisture-ml/full-dataloader-middle')
    )
    input_size = template.x_tensor.size()[2]

    train_data = StreamflowTrainDatasetXY(
        path_csv=f'/data/ssd2/soil-moisture-ml/split-definitions-middle/train_{site_number}.csv',
        dim_1=template.x_tensor.shape[1],
        dim_2=template.x_tensor.shape[2]
    )
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=64)

    test_data = StreamflowTrainDatasetXY_Test(
        path_csv=f'/data/ssd2/soil-moisture-ml/split-definitions-middle/validation_for_train_{site_number}.csv',
        dim_2=template.x_tensor.shape[2]
    )
    data_loader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    model = uNet(num_classes, input_size, max_channels, dropout).to('cuda')
    pretrained_path = "/data/ssd2/soil-moisture-ml/results-pretrain/uNET_middle_18339_sites_08_09_2025-23_18_00_709581/model.pt"
    model.load_state_dict(torch.load(pretrained_path, map_location='cuda'))
    print(f"\u2714 Loaded pre-trained model from {pretrained_path}")

    # Most interpretable order — shallow (input) to deep
    encoder_layers = ["encoder1", "encoder2", "encoder3", "encoder4"]
    for name, param in model.named_parameters():
        if any(name.startswith(enc) for enc in encoder_layers):
            param.requires_grad = True

    initial_fast_lr = 1e-5
    slow_lr = 1e-7
    lr_switch_epoch = 5
    warmup_epochs = 0
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_fast_lr)

    unfreeze_schedule = {
        5: ["encoder1"],
        10: ["encoder2"],
        15: ["encoder3"],
        20: ["encoder4"],
    }

    unfreeze_schedule = {
        200: ["encoder4"],
    }

    losses = []
    test_kge_df = pd.DataFrame(columns=['epoch', 'median_kge', 'lower_kge', 'upper_kge'])
    test_losses_df = pd.DataFrame(columns=['epoch', 'loss'])
    kge_out = []
    path_out = []

    # Evaluate KGE before training
    model.eval()
    test_kges = []

    for X, y, path in data_loader_test:
        with torch.no_grad():
            outputs = model(X.to('cuda'))
            yhat = outputs.cpu().view(-1).numpy()
            yobs = y.cpu().view(-1).numpy()

            kge = compute_kge(yhat, yobs)

        test_kges.append(kge)

    # Save epoch 0 baseline KGE
    test_kge_df = pd.concat([test_kge_df,
                             pd.DataFrame([[0, np.nanmedian(test_kges), np.nanpercentile(test_kges, 25),
                                            np.nanpercentile(test_kges, 75)]],
                                          columns=['epoch', 'median_kge', 'lower_kge', 'upper_kge'])
                             ], ignore_index=True)

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

                # Compute KGE
                yhat = outputs.cpu().view(-1).numpy()
                yobs = y.cpu().view(-1).numpy()

                kge = compute_kge(yhat, yobs)

                test_kges.append(kge)

                # Compute test loss
                loss = criterion(outputs, y.to('cuda'))
                test_loss_batch.append(loss.item())

            # Log KGE
            test_kge_df = pd.concat([test_kge_df, pd.DataFrame(
                [[epoch, np.nanmedian(test_kges), np.nanpercentile(test_kges, 25), np.nanpercentile(test_kges, 75)]],
                columns=['epoch', 'median_kge', 'lower_kge', 'upper_kge'])], ignore_index=True)

            # Log test loss
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
            plt.ylim(0,0.8)
            plt.tight_layout()
            plt.show()

    out_dir = f'/data/ssd2/soil-moisture-ml/results-middle/{model_name}'
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

    # Final training vs test loss plot
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

site_numbers = ['50','100','150', '200', '250',
                '300', '350', '400', '450', '500', '550',
                '600']
for site_number in site_numbers:
    train_and_evaluate(site_number)
