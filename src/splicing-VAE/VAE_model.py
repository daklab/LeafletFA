# factor_model.py 
# maintainer: Karin Isaev 
# date: 2024-01-26

# purpose:  
#   - define VAE model with pyro using just splicing information
#   - framework based on tutorial from https://pyro.ai/examples/vae.html

import os

import numpy as np

import pyro
import pyro.distributions as dist

#assert pyro.__version__.startswith('1.8.6')
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

import random
import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseProportionDataset(Dataset):
    def __init__(self, junction_counts, intron_cluster_counts):
        self.junction_counts = junction_counts.to_dense()
        self.intron_cluster_counts = intron_cluster_counts.to_dense()
        
        # Avoid division by zero by setting zeros in intron_cluster_counts to 1 temporarily for division.
        # This mask will be used to set proportions to 0 where intron_cluster_counts are 0.
        safe_intron_counts = self.intron_cluster_counts.clone()
        safe_intron_counts[safe_intron_counts == 0] = 1
        
        # Calculate proportions safely, then reset proportions to 0 where intron_cluster_counts are 0.
        self.proportions = self.junction_counts / safe_intron_counts
        self.proportions[self.intron_cluster_counts == 0] = 0

    def __len__(self):
        return self.proportions.size(0)

    def __getitem__(self, idx):
        proportion = self.proportions[idx]
        junction_count = self.junction_counts[idx]
        intron_cluster_count = self.intron_cluster_counts[idx]
        
        return {
            'proportions': proportion,
            'junction_counts': junction_count,
            'intron_cluster_counts': intron_cluster_count
        }

def setup_data_loaders(junction_counts, intron_cluster_counts, batch_size=128, use_cuda=False, perc_train=0.6):
    
    # if using cuda move junction tensor and cluster tensor to cuda
    if use_cuda:
        junction_tensor = junction_counts.cuda()
        cluster_tensor = intron_cluster_counts.cuda()
    else:
        junction_tensor = junction_counts
        cluster_tensor = intron_cluster_counts

    dataset = SparseProportionDataset(junction_tensor, cluster_tensor)
    
    # Determine the generator device type based on whether CUDA is used
    generator = torch.Generator(device="cuda" if use_cuda else "cpu")

    # Splitting the dataset into train and test
    print("Percentage of data used for training: ", perc_train)
    train_size = int(perc_train * len(dataset))  
    test_size = len(dataset) - train_size 

    # Make sure random-split is on cpu or else it will be very slow
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device="cuda" if use_cuda else "cpu")) #, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device="cuda" if use_cuda else "cpu")) #, **kwargs)
    
    # also return full dataset for evaluation
    full_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device="cuda" if use_cuda else "cpu")) #, **kwargs)
    
    return train_loader, test_loader, full_data_loader

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, dropout_rate=0.0):
        super().__init__()
        
        modules = []
        # Input layer
        modules.append(nn.Linear(input_dim, hidden_dims[0]))
        #modules.append(nn.BatchNorm1d(hidden_dims[0]))  # Batch normalization
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout_rate))  # Dropout

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            modules.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            #modules.append(nn.BatchNorm1d(hidden_dims[i]))  # Batch normalization
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout_rate))  # Dropout

        self.body = nn.Sequential(*modules)
        
        # Output layers for mean and log variance
        self.linear_means = nn.Linear(hidden_dims[-1], z_dim)
        self.linear_log_var = nn.Linear(hidden_dims[-1], z_dim)

    def forward(self, x):
        x = self.body(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim, dropout_rate=0.0):
        super().__init__()
        
        modules = []
        # Start with a linear layer that maps from the latent space dimension (z_dim) to the first hidden layer dimension.
        modules.append(nn.Linear(z_dim, hidden_dims[-1]))
        #modules.append(nn.BatchNorm1d(hidden_dims[-1]))  # Batch normalization
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout_rate))  # Dropout

        # Iterate over the hidden_dims in reverse order, adding a linear layer, batch normalization, ReLU, and dropout for each.
        for i in range(len(hidden_dims)-2, -1, -1):
            modules.append(nn.Linear(hidden_dims[i+1], hidden_dims[i]))
            #modules.append(nn.BatchNorm1d(hidden_dims[i]))  # Batch normalization
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout_rate))  # Dropout
        
        # The final linear layer transforms the output of the last hidden layer to the desired output dimension.
        modules.append(nn.Linear(hidden_dims[0], output_dim))
        # Note: Do not apply sigmoid here; return logits directly for numerical stability in loss calculation.

        # Wrap the defined layers into a sequential module for simplicity.
        self.body = nn.Sequential(*modules)

    def forward(self, z):
        # Forward pass: given a latent representation z, compute the reconstructed data.
        reconstruction = self.body(z)
        # Return the reconstructed data in logits form.
        return reconstruction
    
def binomial_loss_stable(y_true_counts, n_cluster_counts, logits, eps=1e-04):
    """
    Calculate the binomial loss for a batch of data, focusing on valid data points where cluster counts are greater than 0.

    Args:
        y_true_counts (Tensor): A tensor containing the true junction counts for each observation. (coming from dataloader).
        logits (Tensor): The logits (unactivated outputs).  
        n_cluster_counts (Tensor): A tensor containing the total possible counts (number of trials) for each observation.
        eps (float, optional): A small value to ensure numerical stability in log calculations and probability bounds. Defaults to 1e-04.

    Returns:
        Tensor: The average binomial loss over all valid data points in the batch.
    """

    # ensure that logits are not infinte
    assert not torch.isinf(logits).any()

    # Use torch.logaddexp for numerical stability
    log1p_exp_logits = torch.logaddexp(torch.zeros_like(logits), logits)
    # assert no nan or inf in log1p_exp_logits
    assert not torch.isnan(log1p_exp_logits).any()
    assert not torch.isinf(log1p_exp_logits).any()

    succ = (y_true_counts * logits) 
    fail = (n_cluster_counts * log1p_exp_logits)

    assert not torch.isnan(succ).any()
    assert not torch.isinf(succ).any()
    assert not torch.isnan(fail).any()
    assert not torch.isinf(fail).any()

    # Calculate the log likelihood
    loglik = (y_true_counts * logits) - (n_cluster_counts * log1p_exp_logits)

    # basic calculation
    # loglik = (y_true_counts * (logits)) - (n_cluster_counts * torch.log(1 + torch.exp(logits)))

    # assert no nan in loglik
    assert not torch.isnan(loglik).any()
    # asser no inf in loglik
    assert not torch.isinf(loglik).any()

    loss = -loglik.mean()
    return loss 


def loss_function(recon_x, x, mu, logvar, n_cluster_counts, eps=1e-04):
    """
    Compute the total loss for a batch, combining the binomial reconstruction loss and the KL divergence.

    Args:
        recon_x (Tensor): The reconstructed data (predicted probabilities) from the decoder.
        x (Tensor): The original input data (true junction counts).
        mu (Tensor): The mean vector from the encoder's latent space representation.
        logvar (Tensor): The log variance vector from the encoder's latent space representation.
        n_cluster_counts (Tensor): The total cluster counts corresponding to the number of trials for each observation.
        eps (float, optional): A small value to ensure numerical stability. Defaults to 1e-04.

    Returns:
        Tensor: The total loss for the batch, combining both reconstruction and KL divergence losses.
    """
    # Calculate the reconstruction loss using the custom binomial loss function
    recon_loss = binomial_loss_stable(x, n_cluster_counts, recon_x, eps)
    
    # Calculate the KL divergence loss for regularization
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Sum the reconstruction loss and KL divergence to get the total loss
    total_loss = recon_loss + kl_div
    return total_loss

#  Package the model and guide in a PyTorch module: 
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    This class defines a VAE model, which consists of an encoder and a decoder. The encoder maps input data
    to a latent space representation, while the decoder reconstructs the input data from this latent representation.
    The VAE introduces a stochastic element in the encoding process, allowing for the generation of new data points
    similar to the input data.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dims (list of int): List specifying the sizes of the hidden layers in both encoder and decoder.
        z_dim (int): Dimensionality of the latent space.
        output_dim (int): Dimensionality of the reconstructed output, which matches the input dimension.
    """
    def __init__(self, input_dim, hidden_dims, z_dim, output_dim):
        super().__init__()  # Simplified super call for Python 3
        self.encoder = Encoder(input_dim, hidden_dims, z_dim)
        self.decoder = Decoder(z_dim, hidden_dims, output_dim)

    def reparameterize(self, mu, logvar, apply_noise=True):
        
        std = torch.exp(0.5 * logvar)
        if apply_noise:
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            print("Noise applied: ", apply_noise)
            return mu  # Return the mean without noise if apply_noise is False


    def forward(self, x):
        """
        Defines the forward pass of the VAE.

        During the forward pass, input data is first encoded to a latent space representation,
        then reparameterized to introduce randomness, and finally decoded back to the data space.

        Args:
            x (Tensor): The input data to the VAE.

        Returns:
            tuple: A tuple containing the reconstructed data, the mean, and the log variance of
            the latent space distribution. These are used in the loss function.
        """
        mu, logvar = self.encoder(x)  # Encode the input to get the mean and log variance of the latent space
        z = self.reparameterize(mu, logvar)  # Reparameterize to sample from the latent space
        return self.decoder(z), mu, logvar  # Decode the sample from the latent space, return reconstruction, mu, and logvar
    
    def extract_latent_variables(self, x, apply_noise):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, apply_noise)
        return z


def train(model, train_loader, optimizer, epoch, log_interval=10):
    """
    Train the VAE model for one epoch.

    Args:
        model (torch.nn.Module): The VAE model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        epoch (int): Current epoch number.
        log_interval (int, optional): How often to log training status. Defaults to 10.

    Returns:
        None
    """
    model.train()  # Set the model to training mode
    train_loss = 0  # Accumulator for the total loss in this epoch
    
    # Iterate over batches of data in the training dataset
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data['proportions'])
        loss = loss_function(recon_batch, data['junction_counts'], mu, logvar, data['intron_cluster_counts'])
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    average_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average training loss: {average_loss:.4f}')
    return average_loss

def evaluate(model, test_loader):
    """
    Evaluate the VAE model.

    Args:
        model (torch.nn.Module): The VAE model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        float: The average loss for the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            recon_batch, mu, logvar = model(data['proportions'])
            loss = loss_function(recon_batch, data['junction_counts'], mu, logvar, data['intron_cluster_counts'])
            test_loss += loss.item()
    average_loss = test_loss / len(test_loader.dataset)
    print(f'====> Evaluation set average loss: {average_loss:.4f}')
    return average_loss

def main(junction_tensor, intron_tensor, BATCH_SIZE, USE_CUDA, NUM_EPOCHS, LEARNING_RATE, HIDDEN_DIMS, Z_DIM, OUTPUT_DIM):
    
    # add docstring
    """
    Main function to train VAE model on splicing data

    """
    
    # Configuration
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    USE_CUDA = torch.cuda.is_available()
    
    INPUT_DIM = junction_tensor.shape[1]
    HIDDEN_DIMS = [128, 64]  # Example dimensions, adjust as needed
    Z_DIM = 20
    OUTPUT_DIM = INPUT_DIM  # Assuming output dimension matches input

    # Data Preparation
    train_loader, test_loader, full_data_loader = setup_data_loaders(junction_tensor, intron_tensor, BATCH_SIZE, USE_CUDA)

    # Model Initialization
    model = VAE(INPUT_DIM, HIDDEN_DIMS, Z_DIM, OUTPUT_DIM)
    if USE_CUDA:
        model.cuda()

    # Optimizer Setup
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # best validation loss 
    best_val_loss = float('inf')
    max_patience = 5
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    # Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}")
        avg_train_loss = train(model, train_loader, optimizer, epoch)
        avg_val_loss = evaluate(model, test_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience = max_patience
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == "__main__":
    main()