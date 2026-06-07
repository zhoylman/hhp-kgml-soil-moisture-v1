import sys
import time
import shutil
import datetime
import os
import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import lr_scheduler
import seaborn as sns
import re
import hydroeval as he
import warnings


# Set matplotlib backend for environments without display
import matplotlib
#matplotlib.use('Agg')

# Append custom module path (this script's own dir, where model_config lives)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *
from define_hyperparams import *

def kge_loss(output, target):
    target_mean = torch.mean(target)
    target_std = torch.std(target)
    output_mean = torch.mean(output)
    output_std = torch.std(output)

    r = torch.corrcoef(torch.stack([output.view(-1), target.view(-1)]))[0, 1]
    alpha = output_std / target_std
    beta = output_mean / target_mean

    kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return -kge

def train_and_evaluate(site_number):
    print(f"Training and evaluating model for site_number: {site_number}")
    model_name = f'uNET-shallow_{site_number}_sites_{datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S_%f")}'
    start_time = time.time()
    torch.cuda.get_device_name(torch.cuda.current_device())

    def get_first_file_path(directory):
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return os.path.join(directory, files[0]) if files else None

    template = StreamflowTrainDatasetTemplate(
        csv_file=get_first_file_path('/data/ssd2/soil-moisture-ml/full-dataloader-pretrain-shallow')
    )

    input_size = template.x_tensor.size()[2]

    soilMoistureData = StreamflowTrainDatasetXY(
        path_csv=f'/data/ssd2/soil-moisture-ml/split-definitions-pretrain-shallow/train_{site_number}.csv',
        dim_1=template.x_tensor.shape[1],
        dim_2=template.x_tensor.shape[2]
    )

    # Updated training dataloader
    data_loader = DataLoader(soilMoistureData, batch_size=1024, shuffle=True, num_workers=64)

    streamflowTestData = StreamflowTrainDatasetXY_Test(
        path_csv='/data/ssd2/soil-moisture-ml/split-definitions-pretrain-shallow/validation_1000.csv',
        dim_2=template.x_tensor.shape[2]
    )

    #  Updated test dataloader: larger batch size, more workers
    data_loader_test = DataLoader(streamflowTestData, batch_size=1024, shuffle=False, num_workers=64)

    BiLSTM_model = uNet(num_classes, input_size, max_channels, dropout).to('cuda')
    optimizer = torch.optim.Adam(BiLSTM_model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    losses = []
    test_losses_df = pd.DataFrame(columns=['epoch', 'loss'])
    test_kge_df = pd.DataFrame(columns=['epoch', 'median_kge', 'lower_kge', 'upper_kge'])
    kge_out = []
    path_out = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        batchloss = []

        for X_batch, y_batch in data_loader:
            outputs = BiLSTM_model(X_batch.to('cuda'))
            optimizer.zero_grad()
            loss = criterion(outputs, y_batch.to('cuda'))
            loss.backward()
            optimizer.step()
            batchloss.append(loss.item())

        losses.append(np.mean(batchloss))
        print(f"Epoch: {epoch}, loss: {np.mean(batchloss):.5f}, runtime: {time.time() - epoch_start_time:.1f}s, LR ratio: {scheduler.get_last_lr()[0] / initial_learning_rate:.4f}")

        if epoch % 10 == 0:
            print('Computing test loss / KGE')
            test_losses = []
            test_kges = []

            for X_batch, y_batch, path_batch in data_loader_test:
                with torch.no_grad():
                    outputs_test = BiLSTM_model(X_batch.to('cuda'))

                outputs_test = outputs_test.cpu().numpy()
                y_batch = y_batch.cpu().numpy()

                for i in range(outputs_test.shape[0]):
                    yhat = outputs_test[i].reshape(-1)
                    yobs = y_batch[i].reshape(-1)

                    if np.std(yobs) > 0 and np.mean(yobs) != 0:
                        r = np.corrcoef(yhat, yobs)[0, 1] if np.std(yhat) > 0 else 0
                        alpha = np.std(yhat) / np.std(yobs)
                        beta = np.mean(yhat) / np.mean(yobs)
                        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
                    else:
                        kge = np.nan

                    test_kges.append(kge)

                # Test loss (still batch-wise)
                test_loss = kge_loss(torch.tensor(outputs_test), torch.tensor(y_batch))
                test_losses.append(test_loss.item())

            temp_test_loss = pd.DataFrame([[epoch, np.nanmean(test_losses)]], columns=['epoch', 'loss'])
            temp_test_kge = pd.DataFrame([[epoch, np.nanmedian(test_kges), np.nanpercentile(test_kges, 25), np.nanpercentile(test_kges, 75)]],
                                         columns=['epoch', 'median_kge', 'lower_kge', 'upper_kge'])

            test_losses_df = pd.concat([test_losses_df, temp_test_loss], ignore_index=True)
            test_kge_df = pd.concat([test_kge_df, temp_test_kge], ignore_index=True)

            plt.figure(figsize=(10, 6))
            for y in np.arange(0.1, 1.0, 0.1):
                plt.axhline(y=y, linestyle='--', color='gray', linewidth=0.5)
            plt.plot(test_kge_df['epoch'], test_kge_df['median_kge'], label='Median KGE', color='green', linewidth=2)
            plt.plot(test_kge_df['epoch'], test_kge_df['lower_kge'], label='25th Percentile', color='red', linewidth=1.5)
            plt.plot(test_kge_df['epoch'], test_kge_df['upper_kge'], label='75th Percentile', color='blue', linewidth=1.5)
            plt.xlabel('Epoch')
            plt.ylabel('KGE')
            plt.title('KGE Performance Over Epochs')
            plt.ylim(0, 1)
            plt.xlim(0, num_epochs)
            plt.grid(True, linestyle=':', linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()

        if epoch >= learning_rate_change_epoch:
            scheduler.step()

    out_dir = f'/data/ssd2/soil-moisture-ml/results-pretrain/{model_name}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/plots/', exist_ok=True)
    os.makedirs(f'{out_dir}/hydrograph_data/', exist_ok=True)
    os.makedirs(f'{out_dir}/scripts/', exist_ok=True)

    torch.save(BiLSTM_model.state_dict(), f'{out_dir}/model.pt')

    # Final Validation Hydrograph Plotting
    for X_batch, y_batch, path_batch in data_loader_test:
        with torch.no_grad():
            outputs_test = BiLSTM_model(X_batch.to('cuda'))

        outputs_test = outputs_test.cpu().numpy()
        y_batch = y_batch.cpu().numpy()

        for i in range(outputs_test.shape[0]):
            yhat = outputs_test[i].reshape(-1)
            yobs = y_batch[i].reshape(-1)
            site_id = re.search(r'basin-group-(.*?)-', path_batch[i]).group(1)

            plotting_data = pd.DataFrame({
                'yhat': yhat.flatten(),
                'yobs': yobs.flatten()
            })

            sns.lineplot(data=plotting_data).set_title(f'Site: {site_id}')
            plt.xlabel("Days")
            plt.ylabel("Soil Moisture")
            plt.savefig(f'{out_dir}/plots/plot_{site_id}.png')
            plt.clf()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                kge = he.kge(yhat.reshape(-1, 1), yobs.reshape(-1, 1))

            kge_value = kge[0].item()
            kge_out.append(kge_value)
            path_out.append(path_batch[i])
            plotting_data['kge'] = kge_value
            plotting_data['path'] = path_batch[i]
            plotting_data.to_csv(f'{out_dir}/hydrograph_data/data_{i}.csv')

    # Summarize KGE
    kge_out = np.array(kge_out, dtype=np.float64)
    hist, bin_edges = np.histogram(kge_out, bins=np.linspace(0, 1, 11))
    plt.bar(bin_edges[:-1], hist, width=0.1)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.savefig(f'{out_dir}/kge_hist.png')
    plt.clf()

    median_kge = np.nanmedian(kge_out)
    kge_sorted = np.sort(kge_out)
    cdf = np.arange(1, len(kge_sorted) + 1) / len(kge_sorted)
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

    runtime = time.time() - start_time
    print(f'Runtime was {runtime / 60:.2f} minutes, or {runtime / 3600:.2f} hours')

    out_ungaged_results = pd.DataFrame({'path': path_out, 'kge': kge_out})
    out_ungaged_results.to_csv(f'{out_dir}/ungaged_kge_results.csv', index=False)

    n_param = sum(p.numel() for p in BiLSTM_model.parameters())
    def hyperparam_log():
        return pd.DataFrame({
            'var': ['model_name', 'max_channels', 'input_size', 'hidden_size', 'batch_size', 'num_layers', 'num_classes', 'linear_nodes', 'initial_learning_rate', 'dropout', 'num_epochs', 'runtime_secs', 'median_kge', 'nparam'],
            'vals': [model_name, max_channels, input_size, hidden_size, batch_size, num_layers, num_classes, linear_nodes, initial_learning_rate, dropout, num_epochs, runtime, median_kge, n_param]
        })

    hyperparam_log().to_csv(f'{out_dir}/model_hyperparams.csv', index=False)

    del BiLSTM_model
    torch.cuda.empty_cache()

site_numbers = ['18893']

#criterion = L1WithWindowedDerivativeAndBiasLoss(lambda_deriv=1, lambda_bias=0.5)

for site_number in site_numbers:
    train_and_evaluate(site_number)
