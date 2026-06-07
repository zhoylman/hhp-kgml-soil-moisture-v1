# Import necessary libraries and modules
import sys            # System-specific parameters and functions
import time           # Time-related functions
import shutil         # High-level file operations
import datetime       # Date and time manipulations
import os             # Operating system interface
import pandas as pd   # Data manipulation and analysis
import numpy as np    # Numerical computing with arrays
import torch          # PyTorch for deep learning
import math           # Mathematical functions
import torch.nn as nn # PyTorch neural network module
import matplotlib.pyplot as plt  # Plotting library
from torch.autograd import Variable  # Automatic differentiation for optimization
from torch.utils.data import DataLoader, Dataset, ConcatDataset  # Data loading utilities in PyTorch
import hydroeval      # Hydrological model evaluation tools
from torch.optim import lr_scheduler  # Learning rate scheduler for optimization
import seaborn as sns # Alternative to matplotlib
import re             # Regular expressions

# Define custom Nash-Sutcliffe Efficiency (NSE) loss function
def nse_loss(output, target):
    nse_loss = 1 - ((torch.sum((target - output) ** 2)) / (torch.sum((target - torch.mean(target)) ** 2)))
    return -nse_loss

# Append custom module path to sys for importing user-defined modules
sys.path.append('/home/zhoylman/soil-moisture-ml/py')
from model_config import *  # Import model configuration

# Define the model name based on current datetime for tracking results
model_name = 'LSTM_100_sites'

# Record the start time for runtime calculation
start_time = time.time()

# Check and confirm the GPU being used
torch.cuda.get_device_name(torch.cuda.current_device())

def get_first_file_path(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return os.path.join(directory, files[0]) if files else None

# Load the template data from the training dataset
template = StreamflowTrainDatasetTemplate(
    csv_file = get_first_file_path(
        '/data/ssd2/soil-moisture-ml/full-dataloader'
    )
)

# Extract the input size (number of features) from the template dataset
input_size = template.x_tensor.size()[2]  # Third dimension of x_tensor

# Load model hyperparameters from an external script
from define_hyperparams import *

# Initialize streamflow training data for input-output pairs
soilMoistureData = StreamflowTrainDatasetXY(
    path_csv = '/data/ssd2/soil-moisture-ml/split-definitions/train_100.csv',
    dim_1 = template.x_tensor.shape[1],
    dim_2 = template.x_tensor.shape[2]
)

# Create a DataLoader for batching and shuffling the training data
data_loader = DataLoader(streamflowData, batch_size = batch_size, shuffle = True, num_workers = 64)

# Initialize streamflow testing data for evaluation
testStreamflowData = StreamflowTrainDatasetXY_Test(
    root_dir = '/data/ssd2/streamflow-ml-data-operational/special-case-dataloader/spatiotemporal-holdout/test-dataloader/',
    dim_2 = template.x_tensor.shape[2]
)

# Create a DataLoader for batching and shuffling the test data
data_loader_test = DataLoader(testStreamflowData, batch_size = 1, shuffle = True, num_workers = 0)

# Define the BiLSTM model with specified hyperparameters
BiLSTM_model = BiLSTM(
    num_classes, input_size, hidden_size, dropout, num_layers, linear_nodes, template.x_tensor.shape[2]
)

# Print the model architecture for verification
print(BiLSTM_model)

# Define the optimizer (Adam) and the learning rate scheduler (Exponential decay)
optimizer = torch.optim.Adam(BiLSTM_model.parameters(), lr = initial_learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)

# Move the model to GPU for faster training
BiLSTM_model = BiLSTM_model.to('cuda')

# Declare variables for storing losses and evaluation metrics
losses = []
test_losses_df = pd.DataFrame(columns = ['epoch', 'loss'])
test_nse_df = pd.DataFrame(columns = ['epoch', 'median_nse', 'lower_nse', 'upper_nse'])

# Start the training loop
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    batchloss = []

    # Iterate through the training data in batches
    for X_batch, y_batch in data_loader:
        outputs = BiLSTM_model.forward(X_batch.to('cuda'))
        optimizer.zero_grad()  # Reset gradients
        loss = criterion(outputs, y_batch.to('cuda'))
        loss.backward()
        optimizer.step()
        batchloss.append(loss.item())

    losses.append(np.mean(batchloss))

    print("Epoch: %d, loss: %1.5f, runtime: %1.1f, learning rate ratio (new/init): %1.4f" %
          (epoch, np.mean(batchloss), (time.time() - epoch_start_time),
           (scheduler.get_last_lr()[0] / initial_learning_rate)))

    # Compute validation loss and NSE every 5 epochs
    if epoch % 5 == 0:
        validation_start_time = time.time()
        print('Computing test loss / NSE')
        test_losses = []
        test_nses = []

        for X_batch, y_batch, path in data_loader_test:
            with torch.no_grad():
                outputs_test = BiLSTM_model.forward(X_batch.to('cuda'))
                test_loss = criterion(outputs_test, y_batch.to('cuda'))
                test_losses.append(test_loss.item())

                NSE_data = pd.DataFrame(outputs_test.cpu().detach().numpy()[0, :, 0], columns=['yhat'])
                NSE_data['yobs'] = y_batch.numpy()[0, :]
                test_nse = hydroeval.evaluator(hydroeval.nse, NSE_data['yhat'], NSE_data['yobs'])
                test_nses.append(test_nse.item())

        temp_test_loss = pd.DataFrame([[epoch, np.mean(test_losses)]], columns = ['epoch', 'loss'])
        temp_test_nse = pd.DataFrame([[epoch, np.median(test_nses),
                                        np.percentile(test_nses, 25), np.percentile(test_nses, 75)]],
                                      columns = ['epoch', 'median_nse', 'lower_nse', 'upper_nse'])
        test_losses_df = pd.concat([test_losses_df, temp_test_loss], ignore_index=True)
        test_nse_df = pd.concat([test_nse_df, temp_test_nse], ignore_index=True)

        if epoch % 10 == 0 and epoch != 0:
            plt.hlines([0.5, 0.6, 0.7, 0.8, 0.9], 0, num_epochs, linestyles='--')
            plt.plot(test_nse_df['epoch'], test_nse_df['median_nse'], label='Median Test NSE', color='green')
            plt.plot(test_nse_df['epoch'], test_nse_df['lower_nse'], label='25th Test NSE', color='red')
            plt.plot(test_nse_df['epoch'], test_nse_df['upper_nse'], label='75th Test NSE', color='blue')
            plt.xlabel("Epoch")
            plt.ylabel("Nash–Sutcliffe Efficiency (NSE)")
            plt.ylim([0.3, 1])
            plt.legend()
            plt.show()

    if epoch >= learning_rate_change_epoch:
        scheduler.step()

'''       END OF TRAINING SEQ     '''

# Define results directory
out_dir = '/data/ssd2/streamflow-ml-data-operational/special-case-dataloader/spatiotemporal-holdout/results/' + model_name
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir + '/plots/', exist_ok=True)
os.makedirs(out_dir + '/hydrograph_data/', exist_ok=True)
os.makedirs(out_dir + '/scripts/', exist_ok=True)

# Save loss and NSE curves
test_losses_df.to_csv(out_dir + '/test_losses.csv', index=False)
test_nse_df.to_csv(out_dir + '/test_nse.csv', index=False)

plt.plot(np.arange(0, len(losses)), losses, label='Train Loss', color='blue')
plt.plot(test_losses_df['epoch'], test_losses_df['loss'], label='Test Loss', color='green')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(out_dir + '/loss_curve.png')
plt.show()

plt.hlines(0.5, 0, num_epochs, linestyle='--')
plt.hlines(0.6, 0, num_epochs, linestyle='--')
plt.hlines(0.7, 0, num_epochs, linestyle='--')
plt.hlines(0.8, 0, num_epochs, linestyle='--')
plt.hlines(0.9, 0, num_epochs, linestyle='--')
plt.plot(test_nse_df['epoch'], test_nse_df['median_nse'], label='Median Test NSE', color='green')
plt.plot(test_nse_df['epoch'], test_nse_df['lower_nse'], label='25th Test NSE', color='red')
plt.plot(test_nse_df['epoch'], test_nse_df['upper_nse'], label='75th Test NSE', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Nash–Sutcliffe Efficiency (NSE)")
plt.legend()
plt.savefig(out_dir + '/nse_curve.png')
plt.show()

dataiter = iter(data_loader_test)
nse_out = np.array([])
path_out = np.array([])

for i in range(len(data_loader_test)):
    with torch.no_grad():
        X, y, path = next(dataiter)
        numbers = re.findall(r'\d+', path[0])
        plotting_data = pd.DataFrame(BiLSTM_model(X.data.to('cuda')).cpu().detach().numpy()[0, :, 0],
                                     columns=['yhat'])
        plotting_data['yobs'] = y.numpy()[0, :]
        p = sns.lineplot(data=plotting_data).set_title(
            'Out-of-Sample HUC10: ' + numbers[1] + ', Year: ' + numbers[2])
        plt.xlabel("Days")
        plt.ylabel("Runoff (mm/d)")
        plt.savefig(out_dir + '/plots/plot_' + numbers[1] + '_' + numbers[2] + '.png')
        plt.clf()
        nse = hydroeval.evaluator(hydroeval.nse, plotting_data['yhat'], plotting_data['yobs'])
        nse_out = np.append(nse_out, nse)
        path_out = np.append(path_out, path)
        plotting_data['nse'] = nse[0]
        plotting_data['path'] = path[0]
        plotting_data.to_csv(out_dir + '/hydrograph_data/data_' + str(i) + '.csv')

nse_out[nse_out < 0] = 0
hist, bin_edges = np.histogram(nse_out, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.bar(bin_edges[:-1], hist, width=0.1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.savefig(out_dir + '/nse_hist.png')
plt.show()

median_nse = np.median(nse_out)
nse_sorted = np.sort(nse_out)
cdf = np.arange(1, len(nse_sorted) + 1) / len(nse_sorted)
median_cdf_value = np.interp(median_nse, nse_sorted, cdf)

plt.figure(figsize=(8, 6))
plt.plot(nse_sorted, cdf, label="CDF")
plt.axvline(x=median_nse, color='r', linestyle='--', label=f'Median: {median_nse:.2f}')
plt.scatter(median_nse, median_cdf_value, color='red', zorder=5)
plt.xlabel("NSE")
plt.ylabel("CDF")
plt.title("Cumulative Distribution Function (CDF)\nSpatio-temporal Holdout NSE - Idaho 2011-2013")
plt.legend()
plt.grid(True)
plt.savefig(out_dir + '/nse_cdf.png')
plt.show()

runtime = (time.time() - start_time)
print('Runtime was', runtime / 60, 'minutes, or', runtime / 3600, 'hours')

out_ungaged_results = pd.DataFrame(np.column_stack((path_out, nse_out)), columns=['path', 'nse'])
out_ungaged_results.to_csv(out_dir + '/ungaged_nse_results.csv', encoding='utf-8', index=False)
torch.save(BiLSTM_model.state_dict(), out_dir + '/model.pt')
n_param = sum(p.numel() for p in BiLSTM_model.parameters())

def hyperparam_log(model_name, input_size, hidden_size, batch_size,
                   num_layers, num_classes, linear_nodes, initial_learning_rate,
                   dropout, num_epochs, runtime, median_nse, nparam):
    out = pd.DataFrame(['model_name', 'input_size', 'hidden_size', 'batch_size',
                        'num_layers', 'num_classes', 'linear_nodes', 'initial_learning_rate',
                        'dropout', 'num_epochs', 'runtime_secs', 'median_nse', 'nparam'],
                       columns=['var'])
    out['vals'] = [model_name, input_size, hidden_size, batch_size,
                   num_layers, num_classes, linear_nodes, initial_learning_rate,
                   dropout, num_epochs, runtime, median_nse, nparam]
    return out

hyperparam_out = hyperparam_log(model_name, input_size, hidden_size, batch_size,
                                num_layers, num_classes, linear_nodes, initial_learning_rate,
                                dropout, num_epochs, runtime, median_nse, n_param)

hyperparam_out.to_csv(out_dir + '/model_hyperparams.csv', encoding='utf-8', index=False)

# Save an example training data CSV file for future reference
shutil.copy(get_first_file_path(
    '/data/ssd2/streamflow-ml-data-operational/special-case-dataloader/spatiotemporal-holdout/train-dataloader/'),
    out_dir + '/example-train.csv')

# Copy the current script file for record keeping
current_script_path = '/home/zhoylman/streamflow-ml-operational/py/4_4-lstm-seq-static-spatiotemporal-holdout.py'
shutil.copy(current_script_path, out_dir + '/scripts/model_script.py')

# Copy the model configuration script for record keeping
current_script_path = '/home/zhoylman/streamflow-ml-operational/py/model_config.py'
shutil.copy(current_script_path, out_dir + '/scripts/model_config.py')