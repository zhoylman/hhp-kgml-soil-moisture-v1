
# Import necessary libraries
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
from torch.autograd import Variable
import torch
import os
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------
# 1D UNet-Style Temporal Model for Sequence Prediction
# ----------------------------------------------------------------------------------
# Architecture Overview:
# - Implements a 1D UNet-style encoder-decoder using Conv1D layers
# - Designed for sequence-to-sequence tasks such as time series regression
# - Employs skip connections between encoder and decoder layers to preserve
#   temporal context and improve gradient flow
# - Uses LeakyReLU for activation and Softplus at the output to enforce positivity
#
# Design Notes:
# - All Conv1D layers use `kernel_size=3` with `padding=1` to preserve sequence length
# - `max_channels` controls the model’s capacity; internal layers scale proportionally
# - Dropout can be enabled between layers via `dropout_layer`
#
# Input/Output Format:
# - Input:  [batch_size, sequence_length, input_features]
# - Output: [batch_size, sequence_length, num_classes]
# ----------------------------------------------------------------------------------

class uNet(nn.Module):
    def __init__(self, num_classes, input_size, max_channels, dropout):
        super(uNet, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout = dropout

        # Dynamically compute channel sizes
        c1 = max_channels // 8
        c2 = max_channels // 4
        c3 = max_channels // 2
        c4 = max_channels

        # Encoder
        self.encoder1 = nn.Conv1d(input_size, c1, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv1d(c2, c3, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv1d(c3, c4, kernel_size=3, padding=1)

        # Decoder
        self.decoder3 = nn.Conv1d(c4, c3, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv1d(c3, c2, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv1d(c2, c1, kernel_size=3, padding=1)
        self.decoder0 = nn.Conv1d(c1, num_classes, kernel_size=3, padding=1)

        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.Softplus = nn.Softplus()

    def forward(self, x):
        # x: [batch, sequence, features] → [batch, features, sequence]
        x = x.permute(0, 2, 1)

        e1 = self.relu(self.encoder1(x))
        e2 = self.relu(self.encoder2(e1))
        e3 = self.relu(self.encoder3(e2))
        e4 = self.relu(self.encoder4(e3))

        d3 = self.relu(self.decoder3(e4)) + e3
        d2 = self.relu(self.decoder2(d3)) + e2
        d1 = self.relu(self.decoder1(d2)) + e1
        out = self.decoder0(d1)

        # [batch, num_classes, sequence] → [batch, sequence, num_classes]
        out = out.permute(0, 2, 1)
        out = self.Softplus(out)

        return out

'''

# ----------------------------------------------------------------------------------
# Generalized 1D UNet-Style Model with Pre-CNN Linear Layers
# ----------------------------------------------------------------------------------
# Adds fully connected (linear) layers before the convolutional encoder to expand
# and transform the input feature space prior to spatial (temporal) modeling.
# This allows the model to perform feature interaction and projection before applying CNNs.
# ----------------------------------------------------------------------------------

class uNet(nn.Module):
    def __init__(self, num_classes, input_size, max_channels, dropout):
        super(uNet, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout = dropout

        # Dynamically compute channel sizes
        c1 = max_channels // 8
        c2 = max_channels // 4
        c3 = max_channels // 2
        c4 = max_channels

        # Linear layers before CNN encoder
        self.prelinear1 = nn.Linear(input_size, input_size * 2)
        self.prelinear2 = nn.Linear(input_size * 2, input_size * 4)

        # Encoder
        self.encoder1 = nn.Conv1d(input_size * 4, c1, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv1d(c2, c3, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv1d(c3, c4, kernel_size=3, padding=1)

        # Decoder
        self.decoder3 = nn.Conv1d(c4, c3, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv1d(c3, c2, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv1d(c2, c1, kernel_size=3, padding=1)
        self.decoder0 = nn.Conv1d(c1, num_classes, kernel_size=3, padding=1)

        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.Softplus = nn.Softplus()

    def forward(self, x):
        # x: [batch, sequence, features]
        b, s, f = x.shape

        # Apply linear layers to each timestep independently
        x = self.relu(self.prelinear1(x))
        x = self.relu(self.prelinear2(x))  # [batch, sequence, input_size*4]

        # Permute to [batch, channels, sequence] for Conv1d
        x = x.permute(0, 2, 1)

        # UNet encoder
        e1 = self.relu(self.encoder1(x))
        e2 = self.relu(self.encoder2(e1))
        e3 = self.relu(self.encoder3(e2))
        e4 = self.relu(self.encoder4(e3))

        # UNet decoder with skip connections
        d3 = self.relu(self.decoder3(e4)) + e3
        d2 = self.relu(self.decoder2(d3)) + e2
        d1 = self.relu(self.decoder1(d2)) + e1
        out = self.decoder0(d1)

        # Final activation and reshaping
        out = out.permute(0, 2, 1)  # [batch, sequence, num_classes]
        out = self.Softplus(out)

        return out


class uNet(nn.Module):
    def __init__(self, num_classes, input_size, max_channels, dropout):
        super(uNet, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout = dropout

        # Dynamically compute channel sizes
        c1 = max_channels // 8
        c2 = max_channels // 4
        c3 = max_channels // 2
        c4 = max_channels

        # Encoder
        self.encoder1 = nn.Conv1d(input_size, c1, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv1d(c2, c3, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv1d(c3, c4, kernel_size=3, padding=1)

        # Decoder
        self.decoder3 = nn.Conv1d(c4, c3, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv1d(c3, c2, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv1d(c2, c1, kernel_size=3, padding=1)
        self.decoder0 = nn.Conv1d(c1, num_classes, kernel_size=3, padding=1)

        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.Softplus = nn.Softplus()

    def forward(self, x):
        # x: [batch, sequence, features] → [batch, features, sequence]
        x = x.permute(0, 2, 1)

        e1 = self.relu(self.encoder1(x))
        e2 = self.relu(self.encoder2(e1))
        e3 = self.relu(self.encoder3(e2))
        e4 = self.relu(self.encoder4(e3))

        d3 = self.relu(self.decoder3(e4)) + e3
        d2 = self.relu(self.decoder2(d3)) + e2
        d1 = self.relu(self.decoder1(d2)) + e1
        out = self.decoder0(d1)

        # [batch, num_classes, sequence] → [batch, sequence, num_classes]
        out = out.permute(0, 2, 1)
        out = self.Softplus(out)

        return out


class BiLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, dropout, num_layers, linear_nodes, seq_length):
        super(BiLSTM, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # CNN Encoder
        self.encoder1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # LSTM path on raw input
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True)

        # CNN Decoder
        self.decoder3 = nn.Conv1d(512, 128, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.decoder0 = nn.Conv1d(32, num_classes, kernel_size=3, padding=1)

        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.Softplus = nn.Softplus()

    def forward(self, x):
        # x: [batch, sequence, features]

        # CNN path
        x_cnn = x.permute(0, 2, 1)  # [batch, features, sequence]
        e1 = self.relu(self.encoder1(x_cnn))
        e2 = self.relu(self.encoder2(e1))
        e3 = self.relu(self.encoder3(e2))
        e4 = self.relu(self.encoder4(e3))  # [batch, 256, sequence]

        # LSTM path on raw input
        lstm_out, _ = self.lstm(x)  # [batch, sequence, 256]
        lstm_out = lstm_out.permute(0, 2, 1)  # [batch, 256, sequence]

        # Concatenate LSTM and CNN features
        merged = torch.cat([e4, lstm_out], dim=1)  # [batch, 512, sequence]

        # Decoder with skip connections
        d3 = self.relu(self.decoder3(merged)) + e3
        d2 = self.relu(self.decoder2(d3)) + e2
        d1 = self.relu(self.decoder1(d2)) + e1
        out = self.decoder0(d1)

        out = out.permute(0, 2, 1)  # [batch, sequence, num_classes]
        out = self.Softplus(out)

        return out

class BiLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, dropout, num_layers, linear_nodes, seq_length):
        super(BiLSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_length = seq_length
        self.linear_nodes = linear_nodes

        linear_nodes2 = linear_nodes // 2
        linear_nodes3 = linear_nodes // 4
        linear_nodes4 = linear_nodes // 8
        linear_nodes5 = linear_nodes // 16
        linear_nodes6 = linear_nodes // 32

        self.prelinear1 = nn.Linear(input_size, hidden_size // 2)
        self.prelinear2 = nn.Linear(hidden_size // 2, hidden_size)

        self.LSTM1 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=False)

        self.LSTM2 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=False)

        self.dropout_layer = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_size, linear_nodes)
        self.linear2 = nn.Linear(linear_nodes, linear_nodes2)
        self.linear3 = nn.Linear(linear_nodes2, linear_nodes3)
        self.linear4 = nn.Linear(linear_nodes3, linear_nodes4)
        self.linear5 = nn.Linear(linear_nodes4, linear_nodes5)
        self.linear6 = nn.Linear(linear_nodes5, linear_nodes6)
        self.final_linear_layer = nn.Linear(linear_nodes6, num_classes)

        self.temporal_smoother = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.temporal_smoother.weight, 1 / 3)

        self.relu = nn.LeakyReLU()
        self.Softplus = nn.Softplus()

    def forward(self, x):
        pre_x = self.prelinear1(x)
        pre_x = self.relu(pre_x)
        pre_x = self.prelinear2(pre_x)
        pre_x = self.relu(pre_x)

        h_0 = Variable(torch.zeros(self.num_layers, pre_x.size(0), self.hidden_size)).to(pre_x.device)
        c_0 = Variable(torch.zeros(self.num_layers, pre_x.size(0), self.hidden_size)).to(pre_x.device)

        output, (hn, cn) = self.LSTM1(pre_x, (h_0, c_0))
        output, (hn, cn) = self.LSTM2(output, (hn, cn))

        out = self.relu(output)

        out = self.linear(out)
        out = self.relu(out)
        out = self.dropout_layer(out)

        out = self.linear2(out)
        out = self.relu(out)
        out = self.dropout_layer(out)

        out = self.linear3(out)
        out = self.relu(out)
        out = self.dropout_layer(out)

        out = self.linear4(out)
        out = self.relu(out)
        out = self.dropout_layer(out)

        out = self.linear5(out)
        out = self.relu(out)
        out = self.dropout_layer(out)

        out = self.linear6(out)
        out = self.relu(out)

        out = self.final_linear_layer(out)

        # Apply learnable temporal smoothing
        out = out.permute(0, 2, 1)  # [batch, features=1, sequence]
        out = self.temporal_smoother(out)
        out = out.permute(0, 2, 1)  # [batch, sequence, features=1]

        out = self.Softplus(out)

        return out

class BiLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, dropout, num_layers, linear_nodes, seq_length):
        super(BiLSTM, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout = dropout

        # Temporal UNet-style encoder
        self.encoder1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Temporal UNet-style decoder
        self.decoder3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.decoder0 = nn.Conv1d(32, num_classes, kernel_size=3, padding=1)

        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.Softplus = nn.Softplus()

    def forward(self, x):
        # x: [batch, sequence, features] → [batch, features, sequence]
        x = x.permute(0, 2, 1)

        e1 = self.relu(self.encoder1(x))
        e2 = self.relu(self.encoder2(e1))
        e3 = self.relu(self.encoder3(e2))
        e4 = self.relu(self.encoder4(e3))

        d3 = self.relu(self.decoder3(e4)) + e3
        d2 = self.relu(self.decoder2(d3)) + e2
        d1 = self.relu(self.decoder1(d2)) + e1
        out = self.decoder0(d1)

        # [batch, num_classes, sequence] → [batch, sequence, num_classes]
        out = out.permute(0, 2, 1)
        out = self.Softplus(out)

        return out
'''

# Dataset class for training streamflow data template
class StreamflowTrainDatasetTemplate(Dataset):
    def __init__(self, csv_file):
        """
        Initialize the dataset from a CSV file.
        Args:
            csv_file: Path to the CSV file containing the training data.
        """
        # Load the streamflow data from CSV
        self.streamflow_data = pd.read_csv(csv_file)

        # Extract input features by dropping unnecessary columns
        self.x_train_temp = self.streamflow_data.drop(['time', 'soil_moisture'], axis=1).values

        # Convert features to PyTorch tensor
        self.x_train_tensor_temp = Variable(torch.tensor(self.x_train_temp).float())

        # Reshape tensor and move to GPU
        self.x_tensor = torch.reshape(self.x_train_tensor_temp,
                                      (1, self.x_train_tensor_temp.shape[0], self.x_train_tensor_temp.shape[1])).to(
            'cuda')


class StreamflowTrainDatasetXY(Dataset):
    def __init__(self, path_csv, dim_1, dim_2):
        """
        Initialize the dataset from a CSV listing data file paths.

        Args:
            path_csv: Path to a CSV file containing a column with file paths.
            dim_1: Dimension 1 (sequence length).
            dim_2: Dimension 2 (number of features).
        """
        self.dim_1 = dim_1
        self.dim_2 = dim_2

        # Read CSV with paths
        self.file_list = pd.read_csv(path_csv)

        # Make sure the paths are absolute (optional)
        self.data_paths = self.file_list['path'].tolist()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        sample = pd.read_csv(data_path)

        # Extract input features (X)
        data = sample.drop(['time', 'soil_moisture'], axis=1).values
        data = Variable(torch.tensor(data).float())
        data = torch.reshape(data, (self.dim_1, self.dim_2))

        # Extract target values (Y)
        target = sample[['soil_moisture']].values
        target = Variable(torch.tensor(target).float())
        target = torch.reshape(target, (self.dim_1, 1))

        return data, target


class StreamflowTrainDatasetXY_Test(Dataset):
    def __init__(self, path_csv, dim_2):
        """
        Initialize the dataset for testing using a CSV of file paths.

        Args:
            path_csv: Path to a CSV file containing a column with file paths.
            dim_2: Number of input features.
        """
        self.dim_2 = dim_2
        self.file_list = pd.read_csv(path_csv)
        self.data_paths = self.file_list['path'].tolist()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        sample = pd.read_csv(data_path)

        # Extract input features (X)
        data = sample.drop(['time', 'soil_moisture'], axis=1).values
        data = Variable(torch.tensor(data).float())
        data = torch.reshape(data, (data.shape[0], self.dim_2))

        # Extract target values (Y)
        target = sample[['soil_moisture']].values
        target = Variable(torch.tensor(target).float())
        target = torch.reshape(target, (target.shape[0], 1))

        return data, target, data_path


class StreamflowInference(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Path to the directory with training data.
        """
        self.root_dir = root_dir
        self.data_paths = sorted(os.path.join(root_dir, x) for x in os.listdir(root_dir) if x.endswith('.parquet'))
        self.data_paths_base = sorted(
            os.path.basename(os.path.join(root_dir, x)) for x in os.listdir(root_dir) if x.endswith('.parquet'))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data_path_base = self.data_paths_base[idx]
        sample = pd.read_parquet(data_path, engine='fastparquet')

        # Separate x and y tensors
        time = sample[['time']].values
        data = sample.drop(['time'], axis=1).values

        # Convert to PyTorch tensors
        data = torch.tensor(data, dtype=torch.float32)

        return data, time, data_path_base

class StreamflowInferenceCSV(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Path to the directory with CSV data.
        """
        self.root_dir = root_dir
        self.data_paths = sorted(os.path.join(root_dir, x) for x in os.listdir(root_dir) if x.endswith('.csv'))
        self.data_paths_base = sorted(os.path.basename(x) for x in self.data_paths)

        # Debug: Print out how many files were found
        print(f"Found {len(self.data_paths)} CSV files in {root_dir}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data_path_base = self.data_paths_base[idx]

        # Read CSV instead of Parquet
        sample = pd.read_csv(data_path)

        # Ensure 'time' column exists
        if 'time' not in sample.columns:
            raise ValueError(f"Missing 'time' column in {data_path}")

        # Drop the 'mm_d_q' column if it exists
        if 'mm_d_q' in sample.columns:
            sample = sample.drop(columns=['mm_d_q'])

        # Separate x and y tensors
        time = sample[['time']].values
        data = sample.drop(columns=['time']).values

        # Convert to PyTorch tensors
        data = torch.tensor(data, dtype=torch.float32)

        return data, time, data_path_base

def collate_fn(batch):
    data, times, paths = zip(*batch)
    # Pad sequences to the same length with padding_value=0
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)

    return data_padded, times, paths

def get_first_file_path(directory):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Return the first file path if available, otherwise return None
    return os.path.join(directory, files[0]) if files else None
