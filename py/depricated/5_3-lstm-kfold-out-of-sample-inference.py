# Import necessary libraries and modules
import sys  # System-specific parameters and functions
import time
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Append custom module path to sys for importing user-defined modules
sys.path.append('/home/zhoylman/streamflow-ml-operational/py')  # Adds custom path to access user-defined modules
from model_config import *  # Import custom model configurations

# Define the base directory where k-fold models are stored
k_fold_dir = '/data/ssd2/streamflow-ml-data-operational/k-fold-results/'
model_prefix = 'BiDirLSTM_Duel_k_fold_'
num_models = 10  # Number of models in k-fold

# Iterate through each model in k-fold validation
for k in range(1, num_models + 1):
    model_path = os.path.join(k_fold_dir, f'{model_prefix}{k}', 'model.pt')
    hyperparams_path = os.path.join(k_fold_dir, f'{model_prefix}{k}', 'model_hyperparams.csv')

    # Define the path to forcing data (input data used for model inference)
    forcing_data = f'/data/ssd2/streamflow-ml-data-operational/k-fold-out-of-sample-full-timeseries/k-fold-{k}'

    output_file_path_base = f'/data/ssd2/streamflow-ml-data-operational/k-fold-out-of-sample-full-timeseries/k-fold-{k}/inference-results'

    # Initialize dataset template for streamflow training
    template = StreamflowTrainDatasetTemplate(
        csv_file=get_first_file_path('/data/ssd2/streamflow-ml-data-operational/k-fold-dataloader/fold_1/test-dataloader'))

    # Initialize the test dataset for inference (making predictions)
    testStreamflowData = StreamflowInferenceCSV(root_dir=os.path.join(forcing_data, 'inference-dataloader'))

    # Create a DataLoader to load the test data in batches for inference
    data_loader_test = DataLoader(testStreamflowData, batch_size=1, shuffle=False, num_workers=64,
                                  collate_fn=collate_fn)

    # Initialize an iterator for the DataLoader
    dataiter = iter(data_loader_test)

    # Create the directory if it doesn't exist
    os.makedirs(output_file_path_base, exist_ok=True)

    # Load model hyperparameters
    model_params = pd.read_csv(hyperparams_path)

    # Define and initialize the model
    BiLSTM_model = BiLSTM(
        num_classes=int(model_params[model_params['var'] == 'num_classes']['vals'].iloc[0]),
        input_size=int(model_params[model_params['var'] == 'input_size']['vals'].iloc[0]),
        hidden_size=int(model_params[model_params['var'] == 'hidden_size']['vals'].iloc[0]),
        dropout=float(model_params[model_params['var'] == 'dropout']['vals'].iloc[0]),
        num_layers=int(model_params[model_params['var'] == 'num_layers']['vals'].iloc[0]),
        linear_nodes=int(model_params[model_params['var'] == 'linear_nodes']['vals'].iloc[0]),
        seq_length=template.x_tensor.shape[2]
    )

    # Load pre-trained weights
    BiLSTM_model.load_state_dict(torch.load(model_path))
    BiLSTM_model.to('cuda')  # Move model to GPU
    BiLSTM_model.eval()  # Set model to evaluation mode

    print(f'Running inference for Model {k}...')

    # Iterate through the test data for inference
    for i in range(len(data_loader_test)):
        with torch.no_grad():  # Disable gradient computation
            if i % 100 == 0 and i != 0:
                print(f'Model {k}, Progress: {i / len(data_loader_test) * 100:.2f}% complete')

            X, time, path = next(dataiter)  # Fetch batch of input data, time, and file path

            # Perform inference on the input data using the current model
            output = BiLSTM_model(X.to('cuda')).cpu().detach().numpy()[0, :, 0]

            # Convert model output to a DataFrame
            inference_data = pd.DataFrame(output, columns=[f'yhat_model_{k}'])  # Store results for this model
            inference_data['time'] = [item[0] for item in time[0]]

            # Define output file path
            output_file_path = os.path.join(forcing_data, f'inference-results', f'inference-results-' + path[0]).replace('.parquet', '.csv')

            # Save inference results
            inference_data.to_csv(output_file_path, index=False)

    print(f'Inference complete for Model {k}.')

print('All models have completed inference.')