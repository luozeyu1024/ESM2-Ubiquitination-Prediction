import pandas as pd
import numpy as np
import gc  # Garbage Collector

# Function to reset state of randomness and clear global variables
def reset_state():
    """Reset the state of the random number generator and clear global variables."""
    np.random.seed(None)  # Reset NumPy random seed
    pd.options.mode.chained_assignment = None  # Default warning setting
    gc.collect()  # Execute garbage collection

# Clear cache and reset state
reset_state()


### Import libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import torch.nn.init as init
import torch.optim as optim
import random


# Define paths
open_path = r'Your path to project'

# Load data
test_features = np.load(open_path + r"\ESM2-Ubiquitination-Prediction\Inference_test_data\ESM2_3B_VAE_100\test_features.npy")
test_info_df = pd.read_csv(open_path + r'\ESM2-Ubiquitination-Prediction\Inference_test_data\ESM2_3B_VAE_100\test_info.csv')


# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


### Define required components
# Define DNN model
class DNNLinearModel(nn.Module):
    
    def __init__(self, input_dim):
        super(DNNLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)


    def forward(self, x):
        x = self.fc1(x)
        return x


# Dataset definition
class SequenceDataset(Dataset):
    def __init__(self, features, info_df):
        self.features = features
        self.info_df = info_df
        self.groups = self.info_df.groupby('ID')

    def __len__(self):
        return len(self.groups)
   
    def __getitem__(self, idx):
        group_key = list(self.groups.groups.keys())[idx]
        group = self.groups.get_group(group_key)
        feature_value = self.features[group.index]
        labels = group['Label'].values

        return feature_value, labels


# Custom collate function for DataLoader
def custom_collate_fn(batch):
    features_list = []
    labels_list = []
 
    for features, labels in batch:
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        features_list.append(features_tensor)
        labels_list.append(labels_tensor)
        
    # Stack features using torch.vstack
    batch_features = torch.vstack(features_list)

    # Concatenate labels using torch.cat
    batch_labels = torch.cat(labels_list)

    return batch_features, batch_labels


# Get activation function
def get_activation_fn():
    return nn.LeakyReLU(inplace=False)  # Disable in-place operation

def get_nonlinearity():
    return 'leaky_relu'


# Validation function
def validate(model, device, val_loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Accumulate loss over all batches

    with torch.no_grad():  # Disable gradient computation
        predictions = []
        targets = []
    
        for features, labels in val_loader:
            # Ensure features and labels are on the correct device
            features = features.float().to(device)
            labels = labels.float().to(device)

            # Forward pass
            outputs = model(features)

            # Adjust labels size to match outputs size
            labels = labels.view_as(outputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Accumulate loss
            total_loss += loss.item()
            # Collect predictions and true labels to compute MCC
            predictions.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    # Compute and return average loss
    avg_loss = total_loss / len(val_loader)
    # Compute and return MCC
    mcc = matthews_corrcoef(targets, [1 if p > 0.5 else 0 for p in predictions])
    return avg_loss, mcc

def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    # Set Python's built-in random module seed
    random.seed(seed)
    
    # Set Numpy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # Set the random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Sets all GPU seeds
    
    # Disable cudnn benchmarking to ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call function to set seed
set_seed(42)


# Read label mapping dictionary
best_params = pd.read_excel(open_path + r"\ESM2-Ubiquitination-Prediction\Model\VAE_100_DNNLinerModel\best_hyperparameters.xlsx")
best_params=dict(zip(best_params.iloc[:,0], best_params.iloc[:,1]))


# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Use optimal hyperparameters to retrain the model
learning_rate = best_params['learning_rate']
weight_decay = best_params['weight_decay']
use_pos_weight = bool(best_params['use_pos_weight'])
activation_fn = get_activation_fn()
nonlinearity = get_nonlinearity()
num_epochs = int(best_params['num_epochs'])


## Independent test set
test_dataset = SequenceDataset(test_features, test_info_df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn, drop_last=False)


# Create model
input_dim = test_features.shape[1]
model = DNNLinearModel(input_dim)
model.load_state_dict(torch.load(open_path + r"\ESM2-Ubiquitination-Prediction\Model\VAE_100_DNNLinerModel\VAE_100_DNNLinerModel.pth"))
model.to(device)


# Define loss function and optimizer
if use_pos_weight:
    pos_weight = torch.tensor([len(test_info_df) / (2 * (test_info_df['Label'] == 1).sum())]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Perform inference
test_loss, test_mcc = validate(model, device, test_loader, criterion)

message = f"Test Loss: {test_loss:.6f}, Test MCC: {test_mcc:.6f}"
print(message)