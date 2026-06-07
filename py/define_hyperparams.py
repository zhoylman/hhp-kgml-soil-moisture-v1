import torch
import torch.nn as nn

# Define basic parameters
batch_size = 32  # Batch size for training
num_epochs = 100  # Number of epochs to train the model

# Define dimensionality hyperparameters
# Encoder/decoder parameters
max_channels = 1536
lstm_hidden_size = 256
hidden_size = 256  # Number of hid den units in LSTM (set to 1024 for this model)
num_layers = 1  # Number of stacked LSTM layers (manually managed)
num_classes = 1  # Output size (set to 1 for regression task)
linear_nodes = hidden_size  # Number of nodes in the linear layer (same as hidden size)

# Define regularization parameters
dropout = 0.1  # Dropout rate to prevent overfitting

# Define learning rate and optimizer scheduler
initial_learning_rate = 0.000002  # Initial learning rate for optimizer
final_learning_rate = 0.00000001  # Final learning rate after decay
learning_rate_change_epoch = int(num_epochs * (3/5))  # Epoch at which learning rate starts decaying

# Exponential learning rate scheduler
gamma = (final_learning_rate / initial_learning_rate) ** (1 / (num_epochs - learning_rate_change_epoch))
# 'gamma' is the factor by which the learning rate decays after 'learning_rate_change_epoch'

# Define the loss function (L1 Loss is commonly used for regression tasks)
#criterion = torch.nn.L1Loss()

# ------------------------------------------------------------------------------
# Custom Loss Function: L1WithAsymmetricDerivativeLoss
# ------------------------------------------------------------------------------
# This loss combines standard L1 loss with a derivative-based penalty,
# allowing different weights for rising (wetting) vs falling (drying) trends.
#
# Motivation:
# - Standard L1/L2 losses only penalize prediction value errors.
# - For time series like soil moisture, **sharp increases** from rainfall events
#   occur quickly, while **decreases** happen more gradually.
# - Penalizing prediction slope errors equally in both directions may not be ideal.
#
# How it works:
# - Computes standard L1 loss between prediction and target.
# - Adds a second L1 loss on the **first derivative** (difference across time),
#   with **asymmetric weighting**:
#     - `pos_weight` emphasizes wetting events (positive derivative).
#     - `neg_weight` emphasizes drying (negative derivative).
#
# Benefits:
# - Captures realistic soil moisture dynamics (fast up, slow down).
# - Preserves sharp rainfall responses without over-penalizing smooth drying.
# - Helps the model stay physically plausible while being expressive.
#
# lambda_deriv:
# - Controls how much derivative behavior is enforced relative to prediction error.
# - Higher = stronger focus on matching rate-of-change.
#
# pos_weight / neg_weight:
# - Allows tuning the balance between wetting and drying penalties.
# ------------------------------------------------------------------------------
class L1WithAsymmetricDerivativeLoss(nn.Module):
    def __init__(self, lambda_deriv=0.1, pos_weight=2.0, neg_weight=1.0):
        super(L1WithAsymmetricDerivativeLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.lambda_deriv = lambda_deriv
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred, target):
        loss_main = self.l1(pred, target)

        # Compute time derivatives
        pred_deriv = pred[:, 1:, :] - pred[:, :-1, :]
        target_deriv = target[:, 1:, :] - target[:, :-1, :]
        deriv_error = torch.abs(pred_deriv - target_deriv)

        # Apply asymmetric weights
        pos_mask = (target_deriv > 0).float()
        neg_mask = (target_deriv <= 0).float()
        weight_mask = self.pos_weight * pos_mask + self.neg_weight * neg_mask

        loss_deriv = torch.mean(weight_mask * deriv_error)

        return loss_main + self.lambda_deriv * loss_deriv

# ------------------------------------------------------------------------------
# Custom Loss Function: L1WithWindowedDerivativeLoss
# ------------------------------------------------------------------------------
# Combines standard L1 loss with a **lagged, direction-aware derivative penalty**.
#
# Motivation:
# - For daily soil moisture data, it's important to match not just values but dynamics.
# - Specifically, **wetting** (moisture increase from precipitation) tends to be rapid,
#   while **drying** (evapotranspiration-driven recession) is slower and more gradual.
#
# How it works:
# 1. **L1 Value Loss**:
#    - Penalizes differences between predicted and observed soil moisture values.
#
# 2. **Windowed Derivative Loss**:
#    - Computes the rate of change between time `t` and `t - lag` (e.g., 3 days).
#    - Measures how well the model captures changes over multi-day windows.
#    - **Asymmetric weighting**:
#        - Errors during **wetting events** (positive observed derivatives) are weighted more heavily
#          using `pos_weight` (e.g., 2.0), encouraging sharp model response to rain events.
#        - Errors during **drying events** (negative or flat derivatives) are weighted less
#          using `neg_weight` (e.g., 1.0), allowing the model to smooth these naturally.
#
# Why this matters:
# - Standard L1/L2 losses often **over-smooth** spikes.
# - This formulation pushes the model to **sharpen rising limbs** (more reactive to storms),
#   while **not over-penalizing drying errors**, reflecting real hydrologic processes.
#
# Parameters:
# - lambda_deriv: Controls strength of derivative penalty.
# - deriv_lag: Number of timesteps used to calculate derivative (e.g., 3 = 3-day slope).
# - pos_weight: Emphasis on positive derivatives (e.g., rain-induced wetting).
# - neg_weight: Emphasis on negative or flat derivatives (e.g., drying).
#
# Use this when:
# - You want the model to capture **event-driven variability**, not just trend.
# - You care about **temporal sharpness**, such as rainfall-runoff or moisture recharge events.
# ------------------------------------------------------------------------------

class L1WithWindowedDerivativeLoss(nn.Module):
    def __init__(self, lambda_deriv=0.1, deriv_lag=3, pos_weight=2.0, neg_weight=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_deriv = lambda_deriv
        self.deriv_lag = deriv_lag
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred, target):
        # Standard L1 loss on actual values
        loss_main = self.l1(pred, target)

        # Compute windowed derivatives (t - t-lag)
        pred_deriv = pred[:, self.deriv_lag:, :] - pred[:, :-self.deriv_lag, :]
        target_deriv = target[:, self.deriv_lag:, :] - target[:, :-self.deriv_lag, :]

        # Absolute difference between prediction and target derivatives
        # try squared derivitive loss
        deriv_error = torch.abs(pred_deriv - target_deriv)

        # Direction-aware asymmetric weighting
        pos_mask = (target_deriv > 0).float()
        neg_mask = (target_deriv <= 0).float()
        weight_mask = self.pos_weight * pos_mask + self.neg_weight * neg_mask

        # Weighted derivative loss
        loss_deriv = torch.mean(weight_mask * deriv_error)

        # Total loss: value + derivative components
        return loss_main + self.lambda_deriv * loss_deriv

#criterion = L1WithWindowedDerivativeLoss(lambda_deriv=1)


class L1WithBiasLoss(nn.Module):
    def __init__(self, lambda_bias=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_bias = lambda_bias

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        bias_loss = torch.abs(pred.mean() - target.mean())
        return l1 + self.lambda_bias * bias_loss

#criterion = L1WithBiasLoss(lambda_bias=5.0)

class L1WithWindowedDerivativeAndBiasLoss(nn.Module):
    def __init__(self, lambda_deriv=0.1, lambda_bias=0.2, deriv_lag=3, pos_weight=2.0, neg_weight=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_deriv = lambda_deriv
        self.lambda_bias = lambda_bias
        self.deriv_lag = deriv_lag
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred, target):
        # Main L1 loss
        loss_main = self.l1(pred, target)

        # Derivative loss
        pred_deriv = pred[:, self.deriv_lag:, :] - pred[:, :-self.deriv_lag, :]
        target_deriv = target[:, self.deriv_lag:, :] - target[:, :-self.deriv_lag, :]
        deriv_error = torch.abs(pred_deriv - target_deriv)

        pos_mask = (target_deriv > 0).float()
        neg_mask = (target_deriv <= 0).float()
        weight_mask = self.pos_weight * pos_mask + self.neg_weight * neg_mask
        loss_deriv = torch.mean(weight_mask * deriv_error)

        # Bias penalty
        bias = torch.mean(pred - target)
        loss_bias = torch.abs(bias)

        # Total loss
        return loss_main + self.lambda_deriv * loss_deriv + self.lambda_bias * loss_bias

criterion = L1WithWindowedDerivativeAndBiasLoss(lambda_deriv=1, lambda_bias=0.5)

'''
class L1WithDerivativeLoss(nn.Module):
    def __init__(self, lambda_deriv=0.1):
        super(L1WithDerivativeLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.lambda_deriv = lambda_deriv

    def forward(self, pred, target):
        loss_main = self.l1(pred, target)
        # Compute time derivatives (differences along sequence axis)
        pred_deriv = pred[:, 1:, :] - pred[:, :-1, :]
        target_deriv = target[:, 1:, :] - target[:, :-1, :]
        loss_deriv = self.l1(pred_deriv, target_deriv)
        return loss_main + self.lambda_deriv * loss_deriv

criterion = L1WithDerivativeLoss(lambda_deriv=2)
'''