# Import necessary libraries and modules
import sys  # System-specific parameters and functions
import time  # Time-related functions
import os  # Operating system interface for file and directory manipulations
import torch  # PyTorch for deep learning operations and tensor manipulations
import pandas as pd  # Data manipulation and analysis using DataFrames
import numpy as np  # Numerical computing with arrays, useful for mathematical operations
from torch.utils.data import DataLoader  # PyTorch utility for handling data loading

def get_device_index_by_name(target_name):
    """
    Get the device index of the GPU with the specified name.

    :param target_name: The exact name of the GPU (as returned by torch.cuda.get_device_name()).
    :return: The index of the GPU with the specified name, or None if not found.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    for i in range(torch.cuda.device_count()):
        if target_name.lower() in torch.cuda.get_device_name(i).lower():
            return i
    return None

gpu_name = "NVIDIA RTX 2000E Ada Generation"  # intended inference GPU
device_index = get_device_index_by_name(gpu_name)

# set the inference to run on NVIDIA RTX 2000E Ada Generation (inference GPU)
torch.cuda.set_device(device_index)

# Append custom module path to sys for importing user-defined modules
sys.path.append('/home/zhoylman/streamflow-ml-operational/py')  # Adds custom path to access user-defined modules
from model_config import *  # Import custom model configurations

# Define base directory for models and inference data
k_fold_dir = '/data/ssd2/streamflow-ml-data-operational/k-fold-results/'
model_prefix = 'BiDirLSTM_Duel_k_fold_'
num_models = 10  # Number of models in k-fold
forcing_data = '/data/ssd2/streamflow-ml-data-operational/operational-forcing-data/current'

# Initialize dataset template for streamflow training
template = StreamflowTrainDatasetTemplate(
    csv_file=get_first_file_path('/data/ssd2/streamflow-ml-data-operational/full-dataloader'))

# Initialize the test dataset for inference
testStreamflowData = StreamflowInference(root_dir=os.path.join(forcing_data, 'inference-dataloader'))

# Create a DataLoader to load the test data in batches for inference
data_loader_test = DataLoader(testStreamflowData, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Initialize an iterator for the DataLoader
dataiter = iter(data_loader_test)

# Iterate through each model in k-fold ensemble
for k in range(1, num_models + 1):
    model_path = os.path.join(k_fold_dir, f'{model_prefix}{k}', 'model.pt')
    hyperparams_path = os.path.join(k_fold_dir, f'{model_prefix}{k}', 'model_hyperparams.csv')
    output_file_path_base = os.path.join(forcing_data, f'inference-results-fold-{k}')

    # Create the directory if it doesn't exist
    os.makedirs(output_file_path_base, exist_ok=True)

    # Load model hyperparameters
    model_params = pd.read_csv(hyperparams_path)

    # Define and initialize the BiLSTM model
    BiLSTM_model = BiLSTM(
        num_classes=int(model_params[model_params['var'] == 'num_classes']['vals'].iloc[0]),
        input_size=int(model_params[model_params['var'] == 'input_size']['vals'].iloc[0]),
        hidden_size=int(model_params[model_params['var'] == 'hidden_size']['vals'].iloc[0]),
        dropout=float(model_params[model_params['var'] == 'dropout']['vals'].iloc[0]),
        num_layers=int(model_params[model_params['var'] == 'num_layers']['vals'].iloc[0]),
        linear_nodes=int(model_params[model_params['var'] == 'linear_nodes']['vals'].iloc[0]),
        seq_length=template.x_tensor.shape[2]
    )

    # Load pre-trained model weights
    BiLSTM_model.load_state_dict(torch.load(model_path))
    BiLSTM_model.to('cuda')  # Move model to GPU
    BiLSTM_model.eval()  # Set model to evaluation mode

    print(f'Running inference for Model {k}...')

    # Reset data iterator
    dataiter = iter(data_loader_test)

    # Iterate through the test data for inference
    for i in range(len(data_loader_test)):
        with torch.no_grad():  # Disable gradient computation
            if i % 1000 == 0 and i != 0:
                print(f'Model {k}, Progress: {i / len(data_loader_test) * 100:.2f}% complete')

            X, time, path = next(dataiter)  # Fetch batch of input data, time, and file path

            # Perform inference on the input data using the current model
            output = BiLSTM_model(X.to('cuda')).cpu().detach().numpy()[0, :, 0]

            # Convert model output to a DataFrame
            inference_data = pd.DataFrame(output, columns=[f'yhat_model_{k}'])  # Store results for this model
            inference_data['time'] = [item[0] for item in time[0]]

            # Define output file path
            output_file_path = os.path.join(output_file_path_base, f'inference-results-' + path[0]).replace('.parquet', '.csv')

            # Save inference results
            inference_data.to_csv(output_file_path, index=False)

    print(f'Inference complete for Model {k}.')

print('All models have completed inference.')