import pandas as pd
import numpy as np
import gc  # Garbage Collector

def reset_state():
    """Reset the state of random number generators and clear global variables."""
    np.random.seed(None)  # Reset NumPy random seed
    pd.options.mode.chained_assignment = None  # Default warning settings
    gc.collect()  # Perform garbage collection

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
test_features = np.load(open_path + r"\ESM2-Ubiquitination-Prediction\Inference_test_data\ESM2_3B_2560\test_features.npy")
test_info_df = pd.read_csv(open_path + r'\ESM2-Ubiquitination-Prediction\Inference_test_data\ESM2_3B_2560\test_info.csv')


# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class ContinuousResidualVAE(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.dropout = nn.Dropout(0.3)
            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
            else:
                self.downsample = None

        def forward(self, x):
            out = F.leaky_relu(self.bn(self.fc(x)))
            out = self.dropout(out)
            if self.downsample is not None:
                x = self.downsample(x)
            return out + x
    

    def __init__(self, input_dim, hidden_dim=1280, z_dim=100, loss_type='RMSE', reduction='sum'):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.resblock1 = self.ResBlock(hidden_dim, hidden_dim // 2)
        self.resblock2 = self.ResBlock(hidden_dim // 2, hidden_dim // 4)
        # Latent space
        self.fc21 = nn.Linear(hidden_dim // 4, z_dim)  # Mean layer
        self.fc22 = nn.Linear(hidden_dim // 4, z_dim)  # Log variance layer
        # Decoder
        self.fc3 = nn.Linear(z_dim, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout3 = nn.Dropout(0.3)
        self.resblock3 = self.ResBlock(hidden_dim // 4, hidden_dim // 2)
        self.resblock4 = self.ResBlock(hidden_dim // 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        # Add attributes for loss type and reduction type
        self.loss_type = loss_type
        self.reduction = reduction
        
        if reduction not in ['mean', 'sum']:
            raise ValueError("Invalid reduction type. Expected 'mean' or 'sum', but got %s" % reduction)


    def encode(self, x):
        h = F.leaky_relu(self.bn1(self.fc1(x)))
        h = self.dropout1(h)
        h = self.resblock1(h)
        h = self.resblock2(h)
        return self.fc21(h), self.fc22(h)  # Mean and log variance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from a normal distribution with the same shape as std, mean 0 and standard deviation 1
        return mu + eps * std

    def decode(self, z):
        h = F.leaky_relu(self.bn3(self.fc3(z)))
        h = self.dropout3(h)
        h = self.resblock3(h)
        h = self.resblock4(h)
        return self.fc4(h)  # No sigmoid here

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        
        if self.loss_type == 'MSE':
            self.REC = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction)
        elif self.loss_type == 'RMSE':
            self.REC = torch.sqrt(F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction))
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

        if self.reduction == 'mean':
            self.KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else: 
            self.KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return beta * self.REC + self.KLD
    


    def print_neurons(self):
        print("Encoder neurons:")
        print(f"Input: {self.fc1.in_features}, Output: {self.fc1.out_features}")
        print(f"ResBlock1 - Input: {self.resblock1.fc.in_features}, Output: {self.resblock1.fc.out_features}")
        print(f"ResBlock2 - Input: {self.resblock2.fc.in_features}, Output: {self.resblock2.fc.out_features}")
        
        print("Latent neurons:")
        print(f"Mean (mu) - Input: {self.fc21.in_features}, Output: {self.fc21.out_features}")
        print(f"Log variance (logvar) - Input: {self.fc22.in_features}, Output: {self.fc22.out_features}")

        print("Decoder neurons:")
        print(f"Input: {self.fc3.in_features}, Output: {self.fc3.out_features}")
        print(f"ResBlock3 - Input: {self.resblock3.fc.in_features}, Output: {self.resblock3.fc.out_features}")
        print(f"ResBlock4 - Input: {self.resblock4.fc.in_features}, Output: {self.resblock4.fc.out_features}")
        print(f"Output: {self.fc4.in_features}, Output: {self.fc4.out_features}")

    def get_model_inference_z(self, x, seed=None):
        """
        This function takes input x and returns the corresponding latent vectors z.
        If a seed is provided, it is used to make the random number generator deterministic.
        """
        self.eval()  # Switch to evaluation mode
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():  # Disable gradient computation
            mu, logvar = self.encode(x.view(-1, x.shape[1]))
            z = self.reparameterize(mu, logvar)
        return z


class ResDNNModel(nn.Module):
    
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super(ResDNNModel.ResBlock, self).__init__()
            self.fc1 = nn.Linear(in_dim, out_dim)
            init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
            self.bn1 = nn.BatchNorm1d(out_dim)
            
            self.fc2 = nn.Linear(out_dim, out_dim)
            init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
            self.bn2 = nn.BatchNorm1d(out_dim)
            
            self.activation_fn = get_activation_fn()
            
            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            identity = x
            out = self.activation_fn(self.bn1(self.fc1(x)))
            out = self.activation_fn(self.bn2(self.fc2(out)))
            
            if self.downsample is not None:
                identity = self.downsample(identity)
                
            out += identity
            return out
   
    def __init__(self, input_dim, layer_dims):
        super(ResDNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dims[0])
        init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        self.bn1 = nn.BatchNorm1d(layer_dims[0])
        
        self.resblocks = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.resblocks.append(self.ResBlock(layer_dims[i], layer_dims[i+1]))
        
        self.fc_out = nn.Linear(layer_dims[-1], 1)
        init.kaiming_normal_(self.fc_out.weight, nonlinearity='sigmoid')

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=False)  # Disable in-place operation
        for block in self.resblocks:
            x = block(x)
        x = self.fc_out(x)
        return x


# Create combined model
class CombinedModel(nn.Module):
    def __init__(self, vae, dnn):
        super(CombinedModel, self).__init__()
        self.vae = vae
        self.dnn = dnn

    def forward(self, x):
        mu, logvar  = self.vae.encode(x)  # Use only the encoder of VAE
        latent = self.vae.reparameterize(mu, logvar)
        return self.dnn(latent)


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


def get_activation_fn():
    return nn.LeakyReLU(inplace=False)  # Disable in-place operation
    # return nn.LeakyReLU()


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

            # Forward propagation
            outputs = model(features)

            # Adjust labels to match the dimensions of outputs
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


# Load VAE model
input_dim = 2560
hidden_dim = 1280
z_dim = 100
vae_model = ContinuousResidualVAE(input_dim, hidden_dim, z_dim)
# Load DNN model
best_params = pd.read_excel(open_path + r"\ESM2-Ubiquitination-Prediction\Model\VAE_ResDNNModel\VAE_ResDNNModel_best_hyperparameters.xlsx")
best_params = dict(zip(best_params.iloc[:,0], best_params.iloc[:,1]))
use_pos_weight = bool(best_params['use_pos_weight'])
num_blocks = best_params['num_blocks']
layer_dims = [int(best_params[f'layer_{i}_dim']) for i in range(int(num_blocks) + 1)]
input_dim = 100
vae_100_resdnnmodel = ResDNNModel(input_dim, layer_dims)
# Combined model
vae_resdnn_model = CombinedModel(vae_model, vae_100_resdnnmodel)
# Load the combined model
vae_resdnn_model.load_state_dict(torch.load(open_path + r"\ESM2-Ubiquitination-Prediction\Model\VAE_ResDNNModel\VAE_ResDNNModel.pth"))
vae_resdnn_model.to(device)


# Independent test set
test_dataset = SequenceDataset(test_features, test_info_df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn, drop_last=False)


# Define loss function and optimizer
if use_pos_weight:
    pos_weight = torch.tensor([len(test_info_df) / (2 * (test_info_df['Label'] == 1).sum())]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.BCEWithLogitsLoss()


# Perform inference
test_loss, test_mcc = validate(vae_resdnn_model, device, test_loader, criterion)

# Record end time
message = f"Test Loss: {test_loss:.6f}, Test MCC: {test_mcc:.6f}"
print(message)
