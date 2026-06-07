# Import necessary libraries and modules
import sys  # System-specific parameters and functions
import time  # Time-related functions
import shutil  # High-level file operations
import datetime  # Date and time manipulations
import os  # Operating system interface
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing with arrays
import torch  # PyTorch for deep learning
import math  # Mathematical functions
import torch.nn as nn  # PyTorch neural network module
import matplotlib.pyplot as plt  # Plotting library
from torch.autograd import Variable  # Automatic differentiation for optimization
from torch.utils.data import DataLoader, Dataset, ConcatDataset  # Data loading utilities in PyTorch
import hydroeval  # Hydrological model evaluation tools
from torch.optim import lr_scheduler  # Learning rate scheduler for optimization
import shutil  # For copying and removing files
import seaborn as sns # Alternative to matplotlib
import re

# Define custom Nash-Sutcliffe Efficiency (NSE) loss function
def nse_loss(output, target):
    nse_loss = 1 - ((torch.sum((target - output) ** 2)) / (torch.sum((target - torch.mean(target)) ** 2)))
    return -nse_loss

for f in range(1, 10+1):
    # Append custom module path to sys for importing user-defined modules
    sys.path.append('/home/zhoylman/streamflow-ml-operational/py')
    from model_config import *  # Import model configuration

    # Define the model name based on current datetime for tracking results
    model_name = 'BiDirLSTM_Duel_k_fold_' + str(f)

    # Record the start time for runtime calculation
    start_time = time.time()

    # Check and confirm the GPU being used
    torch.cuda.get_device_name(torch.cuda.current_device())

    def get_first_file_path(directory):
        # List all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        # Return the first file path if available, otherwise return None
        return os.path.join(directory, files[0]) if files else None


    # Load the template data from the training dataset
    template = StreamflowTrainDatasetTemplate(
        csv_file=get_first_file_path(
            '/data/ssd2/streamflow-ml-data-operational/k-fold-dataloader/fold_' + str(f) + '/train-dataloader/'))

    # Extract the input size (number of features) from the template dataset
    input_size = template.x_tensor.size()[2]  # Input size is the third dimension of x_tensor

    # Load model hyperparameters from an external script
    from define_hyperparams import *

    # Initialize streamflow training data for input-output pairs
    streamflowData = StreamflowTrainDatasetXY(
        root_dir='/data/ssd2/streamflow-ml-data-operational/k-fold-dataloader/fold_' + str(f) + '/train-dataloader/',
        dim_1=template.x_tensor.shape[1],
        dim_2=template.x_tensor.shape[2]
    )

    # Create a DataLoader for batching and shuffling the training data
    data_loader = DataLoader(streamflowData, batch_size=batch_size, shuffle=True, num_workers=64)

    # Initialize streamflow testing data for evaluation
    testStreamflowData = StreamflowTrainDatasetXY_Test(
        root_dir='/data/ssd2/streamflow-ml-data-operational/k-fold-dataloader/fold_' + str(f) + '/test-dataloader/',
        dim_2=template.x_tensor.shape[2]
    )

    # Create a DataLoader for batching and shuffling the test data
    data_loader_test = DataLoader(testStreamflowData, batch_size=1, shuffle=True, num_workers=0)

    # Define the BiLSTM model with specified hyperparameters
    BiLSTM_model = BiLSTM(
        num_classes, input_size, hidden_size, dropout, num_layers, linear_nodes, template.x_tensor.shape[2]
    )

    # Print the model architecture for verification
    print(BiLSTM_model)

    # Define the optimizer (Adam) and the learning rate scheduler (Exponential decay)
    optimizer = torch.optim.Adam(BiLSTM_model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Move the model to GPU for faster training
    BiLSTM_model = BiLSTM_model.to('cuda')

    # Declare variables for storing losses and evaluation metrics
    losses = []
    test_losses_df = pd.DataFrame(columns=['epoch', 'loss'])
    test_nse_df = pd.DataFrame(columns=['epoch', 'median_nse', 'lower_nse', 'upper_nse'])

    # Start the training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # List to store batch-wise losses
        batchloss = []

        # Iterate through the training data in batches
        for X_batch, y_batch in data_loader:
            # Forward pass through the model
            outputs = BiLSTM_model.forward(X_batch.to('cuda'))
            optimizer.zero_grad()  # Reset gradients

            # Compute loss and perform backpropagation
            loss = criterion(outputs, y_batch.to('cuda'))
            loss.backward()
            optimizer.step()
            batchloss.append(loss.item())

        # Store average batch loss for this epoch
        losses.append(np.mean(batchloss))

        # Print training progress after every epoch
        if epoch % 1 == 0:
            print("Epoch: %d, loss: %1.5f, runtime: %1.1f, learning rate ratio (new/init): %1.4f" %
                  (epoch, np.mean(batchloss), (time.time() - epoch_start_time),
                   (scheduler.get_last_lr()[0] / initial_learning_rate)))

        # Compute validation loss and NSE every 5 epochs
        if epoch % 5 == 0:
            validation_start_time = time.time()
            print('Computing test loss / NSE')
            test_losses = []
            test_nses = []

            # Iterate through the test data
            for X_batch, y_batch, path in data_loader_test:
                with torch.no_grad():
                    # Compute test loss and NSE
                    outputs_test = BiLSTM_model.forward(X_batch.to('cuda'))
                    test_loss = criterion(outputs_test, y_batch.to('cuda'))
                    test_losses.append(test_loss.item())

                    # Prepare data for NSE calculation
                    NSE_data = pd.DataFrame(outputs_test.cpu().detach().numpy()[0, :, 0], columns=['yhat'])
                    NSE_data['yobs'] = y_batch.numpy()[0, :]

                    # Compute NSE
                    test_nse = hydroeval.evaluator(hydroeval.nse, NSE_data['yhat'], NSE_data['yobs'])
                    test_nses.append(test_nse.item())

            # Log validation metrics
            print('Validation runtime: %1.1f' % (time.time() - validation_start_time))
            temp_test_loss = pd.DataFrame([[epoch, np.mean(test_losses)]], columns=['epoch', 'loss'])
            temp_test_nse = pd.DataFrame(
                [[epoch, np.median(test_nses), np.percentile(test_nses, 25), np.percentile(test_nses, 75)]],
                columns=['epoch', 'median_nse', 'lower_nse', 'upper_nse'])
            test_losses_df = pd.concat([test_losses_df, temp_test_loss], ignore_index=True)
            test_nse_df = pd.concat([test_nse_df, temp_test_nse], ignore_index=True)

            # Plot the validation NSE as training progresses (every 10 epochs)
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

        # Step the learning rate scheduler after a certain epoch
        if epoch >= learning_rate_change_epoch:
            scheduler.step()

    '''       END OF TRAINING SEQ     '''


    # Function to log the hyperparameters of the model
    # and the results after training.
    def hyperparam_log(model_name, input_size,
                       hidden_size, batch_size,
                       num_layers, num_classes,
                       linear_nodes, initial_learning_rate,
                       dropout, num_epochs,
                       runtime, median_nse, nparam):
        # Create a DataFrame with variable names and their corresponding values
        out = pd.DataFrame(['model_name', 'input_size',
                            'hidden_size', 'batch_size',
                            'num_layers', 'num_classes',
                            'linear_nodes', 'initial_learning_rate',
                            'dropout', 'num_epochs',
                            'runtime_secs',
                            'median_nse', 'nparam'], columns=['var'])
        # Fill the DataFrame with values of the hyperparameters
        out['vals'] = [model_name, input_size,
                       hidden_size, batch_size,
                       num_layers, num_classes,
                       linear_nodes, initial_learning_rate,
                       dropout, num_epochs,
                       runtime, median_nse, nparam]
        return out


    # If the model runs all the way, make a results folder to store the outputs
    out_dir = '/data/ssd2/streamflow-ml-data-operational/k-fold-results/' + model_name
    os.mkdir(out_dir)  # Create main results directory
    os.mkdir(out_dir + '/plots/')  # Create directory for plots
    os.mkdir(out_dir + '/hydrograph_data/')  # Create directory for hydrograph data
    os.mkdir(out_dir + '/scripts/')  # Create directory to archive scripts

    #write out csvs of loss and NSE curves (training)
    #this will help us to show that the training schemes are the same
    # Save test_losses_df to a CSV file
    test_losses_df.to_csv(out_dir + '/test_losses.csv', index=False)
    # Save test_nse_df to a CSV file
    test_nse_df.to_csv(out_dir + '/test_nse.csv', index=False)

    # Plot the training and test loss curves
    plt.plot(np.arange(0, len(losses)), losses, label='Train Loss', color='blue')  # Training loss
    plt.plot(test_losses_df['epoch'], test_losses_df['loss'], label='Test Loss', color='green')  # Test loss
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_dir + '/loss_curve.png')  # Save the plot as a PNG file
    plt.show()

    # Plot the Nash-Sutcliffe Efficiency (NSE) curve
    plt.hlines(0.5, 0, num_epochs, linestyle='--')  # Reference line at NSE = 0.5
    plt.hlines(0.6, 0, num_epochs, linestyle='--')  # Reference line at NSE = 0.6
    plt.hlines(0.7, 0, num_epochs, linestyle='--')  # Reference line at NSE = 0.7
    plt.hlines(0.8, 0, num_epochs, linestyle='--')  # Reference line at NSE = 0.8
    plt.hlines(0.9, 0, num_epochs, linestyle='--')  # Reference line at NSE = 0.9
    plt.plot(test_nse_df['epoch'], test_nse_df['median_nse'], label='Median Test NSE', color='green')  # Median NSE
    plt.plot(test_nse_df['epoch'], test_nse_df['lower_nse'], label='25th Test NSE', color='red')  # 25th percentile NSE
    plt.plot(test_nse_df['epoch'], test_nse_df['upper_nse'], label='75th Test NSE', color='blue')  # 75th percentile NSE
    plt.xlabel("Epoch")
    plt.ylabel("Nash–Sutcliffe Efficiency (NSE)")
    plt.legend()
    plt.savefig(out_dir + '/nse_curve.png')  # Save the NSE curve plot
    plt.show()

    # Build ungaged basin results
    dataiter = iter(data_loader_test)
    nse_out = np.array([])  # Initialize an empty array to store NSE values
    path_out = np.array([])  # Initialize an empty array to store the file paths of test data

    # Iterate over the test dataset
    for i in range(len(data_loader_test)):
        with torch.no_grad():
            X, y, path = next(dataiter)
            # Extract numbers from path
            numbers = re.findall(r'\d+', path[0])
            # Generate predictions using the trained model (forward pass)
            plotting_data = pd.DataFrame(BiLSTM_model(X.data.to('cuda')).cpu().detach().numpy()[0, :, 0],
                                         columns=['yhat'])  # Predicted streamflow
            plotting_data['yobs'] = y.numpy()[0, :]  # Observed streamflow
            p = sns.lineplot(data=plotting_data).set_title(
                'Out-of-Sample HUC10: ' + numbers[1] + ', Year: ' + numbers[2])  # Plot predictions vs observations
            plt.xlabel("Days")
            plt.ylabel("Runoff (mm/d)")
            plt.savefig(out_dir + '/plots/plot_' + numbers[1] + '_' + numbers[2] + '.png')  # Save the plot as PNG
            plt.clf()  # Clear the current figure for the next plot

            # Calculate Nash-Sutcliffe Efficiency (NSE)
            nse = hydroeval.evaluator(hydroeval.nse, plotting_data['yhat'], plotting_data['yobs'])
            nse_out = np.append(nse_out, nse)  # Store NSE value
            path_out = np.append(path_out, path)  # Store corresponding file path

            plotting_data['nse'] = nse[0]  # Add NSE to the DataFrame
            plotting_data['path'] = path[0]  # Add file path to the DataFrame

            # Write out the prediction and observation data as CSV
            plotting_data.to_csv(out_dir + '/hydrograph_data/data_' + str(i) + '.csv')

    # Plot the histogram of NSE values
    nse_out[nse_out < 0] = 0  # Set negative NSE values to zero
    hist, bin_edges = np.histogram(nse_out, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.bar(bin_edges[:-1], hist, width=0.1)  # Create a bar chart of NSE distribution
    plt.xlim(min(bin_edges), max(bin_edges))  # Set the x-axis limits to the bin range
    plt.savefig(out_dir + '/nse_hist.png')  # Save the histogram plot
    plt.show()

    # compute median NSE
    median_nse = np.median(nse_out)  # Compute the median NSE value

    # Compute the Cumulative Distribution Function (CDF) of the NSE values
    nse_sorted = np.sort(nse_out)  # Sort NSE values in ascending order
    cdf = np.arange(1, len(nse_sorted) + 1) / len(nse_sorted)  # Calculate the CDF

    # Compute the CDF value at the median
    median_cdf_value = np.interp(median_nse, nse_sorted, cdf)

    # Plot the CDF of NSE values
    plt.figure(figsize=(8, 6))
    plt.plot(nse_sorted, cdf, label="CDF")  # Plot the CDF
    plt.axvline(x=median_nse, color='r', linestyle='--',
                label=f'Median: {median_nse:.2f}')  # Add a vertical line at the median
    plt.scatter(median_nse, median_cdf_value, color='red', zorder=5)  # Mark the median point

    plt.xlabel("NSE")
    plt.ylabel("CDF")
    plt.title("Cumulative Distribution Function (CDF) of out-of-sample NSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir + '/nse_cdf.png')  # Save the CDF plot
    plt.show()

    # Compute the runtime of the model training
    runtime = (time.time() - start_time)
    print('Runtime was', runtime / 60, 'minutes, or', runtime / 3600, 'hours')

    # Compute the ungaged results (NSE for each test set)
    out_ungaged_results = pd.DataFrame(np.column_stack((path_out, nse_out)), columns=['path', 'nse'])

    # Write out the ungaged results as CSV
    out_ungaged_results.to_csv(out_dir + '/ungaged_nse_results.csv', encoding='utf-8', index=False)

    # Save the trained BiLSTM model parameters
    torch.save(BiLSTM_model.state_dict(), out_dir + '/model.pt')

    # Calculate the total number of parameters in the model
    n_param = sum(p.numel() for p in BiLSTM_model.parameters())

    # Log the model's hyperparameters and performance
    hyperparam_out = hyperparam_log(model_name, input_size,
                                    hidden_size, batch_size,
                                    num_layers, num_classes,
                                    linear_nodes, initial_learning_rate,
                                    dropout, num_epochs,
                                    runtime, median_nse,
                                    n_param)

    # Write out the hyperparameters as CSV
    hyperparam_out.to_csv(out_dir + '/model_hyperparams.csv', encoding='utf-8', index=False)

    # Finally, save an example training data CSV file for future reference
    shutil.copy(get_first_file_path(
            '/data/ssd2/streamflow-ml-data-operational/k-fold-dataloader/fold_' + str(f) + '/train-dataloader/'),
                out_dir + '/example-train.csv')

    # Path to the current script file
    current_script_path = '/home/zhoylman/streamflow-ml-operational/py/4_3-lstm-seq-static-k-fold.py'  # This gets the current script's file path

    # Destination where the script will be copied to
    destination_path = out_dir + '/scripts/model_script.py'

    # Copy the script file to the results folder
    shutil.copy(current_script_path, destination_path)

    # Path to the current script file
    current_script_path = '/home/zhoylman/streamflow-ml-operational/py/model_config.py'  # This gets the current script's file path

    # Destination where the script will be copied to
    destination_path = out_dir + '/scripts/model_config.py'

    # Copy the script file to the results folder
    shutil.copy(current_script_path, destination_path)