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

#class SparseProportionDataset(Dataset):
#    def __init__(self, junction_counts, intron_cluster_counts):
#        """
#        Initializes the dataset with sparse tensors for proportions, junction counts, and intron cluster counts.
#        proportions: A sparse tensor representing the proportion of successes.
#        junction_counts: A sparse tensor representing the number of successes.
#        intron_cluster_counts: A sparse tensor representing the number of trials.
#        """
#        self.junction_counts = junction_counts.to_dense()
#        self.intron_cluster_counts = intron_cluster_counts.to_dense()
#        self.proportions = self.junction_counts / self.intron_cluster_counts
#
#    def __len__(self):
#        # Assuming all tensors have the same first dimension size
#        return self.proportions.size(0)
#
#    def __getitem__(self, idx):
#        """
#        Retrieves the dense slice of the dataset for the given index. This involves converting the sparse slice
#        into a dense format.
#        """
#        # Convert sparse slice to dense format, if necessary
#        # For very large sparse matrices, consider using batch processing techniques.
#        
#        proportion = self.proportions[idx]
#        junction_count = self.junction_counts[idx]
#        intron_cluster_count = self.intron_cluster_counts[idx]
#        
#        return {
#            'proportions': proportion,
#            'junction_counts': junction_count,
#            'intron_cluster_counts': intron_cluster_count
#        }

def setup_data_loaders(junction_counts, intron_cluster_counts, batch_size=128, use_cuda=False):
    
    # if using cuda move junction tensor and cluster tensor to cuda
    if use_cuda:
        junction_tensor = junction_counts.cuda()
        cluster_tensor = intron_cluster_counts.cuda()

    dataset = SparseProportionDataset(junction_tensor, cluster_tensor)
    
    # Determine the generator device type based on whether CUDA is used
    generator = torch.Generator(device="cuda" if use_cuda else "cpu")

    # Splitting the dataset into train and test
    train_size = int(0.8 * len(dataset))  # 80% training
    test_size = len(dataset) - train_size  # 20% testing

    # Make sure random-split is on cpu or else it will be very slow
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    #kwargs = {'num_workers': 0, 'pin_memory': use_cuda}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device="cuda" if use_cuda else "cpu")) #, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device="cuda" if use_cuda else "cpu")) #, **kwargs)
    return train_loader, test_loader

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__()  # Simplified super call for Python 3
        
        # Build a sequential neural network model dynamically based on the provided list of hidden dimensions.
        # The first layer transforms the input dimension to the first hidden layer dimension.
        modules = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        # Iterate over the list of hidden dimensions to create additional layers.
        # For each layer, create a linear transformation followed by a ReLU activation function,
        # transforming from the current hidden layer dimension to the next one.
        for i in range(len(hidden_dims)-1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            modules.append(nn.ReLU())
        # Wrap the layers into a sequential container to automate forward propagation through these layers.
        self.body = nn.Sequential(*modules)
        
        # After passing through the hidden layers, output two vectors from the final hidden layer dimension:
        # one for the means (μ) and one for the log variances (log(σ^2)) of the latent variables.
        # These will parameterize the Gaussian distributions from which we'll sample the latent variables.
        self.linear_means = nn.Linear(hidden_dims[-1], z_dim)  # Linear transformation for the mean μ of the latent variables.
        self.linear_log_var = nn.Linear(hidden_dims[-1], z_dim)  # Linear transformation for the log variance log(σ^2) of the latent variables.

    def forward(self, x):
        # Forward pass through the initial body of the network
        x = self.body(x)  # Pass the input x through the sequential layers defined in self.body.
        
        # Calculate the mean μ and log variance log(σ^2) for each example in the batch.
        # These will be used in the reparameterization trick to sample from the latent space.
        means = self.linear_means(x)  # Compute the means μ for the latent variables.
        log_vars = self.linear_log_var(x)  # Compute the log variances log(σ^2) for the latent variables.
        
        # Return the mean and log variance of the latent space variables.
        # These are critical for the reparameterization step and subsequent loss calculation.
        return means, log_vars

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim):
        super().__init__()  # Simplified super call for Python 3
        # Initialize the decoder with dimensions for the latent space (z_dim),
        # a list of dimensions for hidden layers (hidden_dims), and the dimensionality
        # of the output data (output_dim).
        
        # Create a reverse sequence of layers from the latent space back to the observed space.
        # Start with a linear layer that maps from the latent space dimension (z_dim) to the
        # dimension of the last hidden layer (the first in the reverse sequence).
        modules = [nn.Linear(z_dim, hidden_dims[-1]), nn.ReLU()]
        
        # Iterate over the hidden_dims in reverse order (except the last one already used),
        # adding a linear layer and a ReLU activation for each. This progressively expands
        # the representation back towards the input dimensionality.
        for i in range(len(hidden_dims)-2, -1, -1):
            modules.append(nn.Linear(hidden_dims[i+1], hidden_dims[i]))
            modules.append(nn.ReLU())
        
        # The final linear layer transforms the output of the last hidden layer
        # to the desired output dimension. No activation is applied here yet, as
        # we typically apply the final activation (e.g., sigmoid for probabilities)
        # outside the model or in the forward pass, depending on the specific use case.
        modules.append(nn.Linear(hidden_dims[0], output_dim))
        
        # Wrap the defined layers into a sequential module for simplicity.
        self.body = nn.Sequential(*modules)

    def forward(self, z):
        # Forward pass for the decoder: given a latent representation z,
        # compute the reconstructed data.
        
        # Pass the latent vector z through the sequence of layers to obtain
        # the reconstruction. The final layer's output is passed through a sigmoid
        # to ensure the output values are in the [0, 1] range, suitable for proportions.
        reconstruction = torch.sigmoid(self.body(z))
        
        # Return the reconstructed data.
        return reconstruction

def binomial_loss(y_true_counts, n_cluster_counts, y_pred_probs, eps=1e-04):
    """
    Calculate the binomial loss for a batch of data, focusing on valid data points where cluster counts are greater than 0.

    Args:
        y_true_counts (Tensor): A tensor containing the true junction counts for each observation. (coming from dataloader)
        y_pred_probs (Tensor): A tensor containing the predicted probabilities of observing each junction count.
        n_cluster_counts (Tensor): A tensor containing the total possible counts (number of trials) for each observation.
        eps (float, optional): A small value to ensure numerical stability in log calculations and probability bounds. Defaults to 1e-04.

    Returns:
        Tensor: The average binomial loss over all valid data points in the batch.
    """
        
    # Ensure predictions are within the valid range [eps, 1-eps] to prevent log(0) issues.
    y_pred_probs = torch.clamp(y_pred_probs, eps, 1 - eps)
    
    # Create a mask to identify valid data points with non-zero cluster counts.
    valid_data_mask = (n_cluster_counts > 0)
    
    # Apply the mask to select valid observations, ignoring NaNs and zeros in the dataset.
    val_junc_counts = y_true_counts[valid_data_mask]
    val_total_counts = n_cluster_counts[valid_data_mask]
    val_y_pred_probs = y_pred_probs[valid_data_mask]

    # Calculate the binomial loss for valid observations using negative log likelihood.
    loss = - (val_junc_counts * torch.log(val_y_pred_probs) + (val_total_counts - val_junc_counts) * torch.log(1 - val_y_pred_probs))
    
    # Normalize the loss by the number of valid data points to get the average loss.
    loss = loss.sum() / valid_data_mask.sum()
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
    recon_loss = binomial_loss(x, n_cluster_counts, recon_x, eps)
    
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

    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick to sample from the latent space.

        This method takes the mean and log variance of the latent space distribution (as output by the encoder),
        and samples from this distribution by adding a scaled random noise, thus enabling backpropagation through
        stochastic nodes.

        Args:
            mu (Tensor): The mean of the latent space distribution.
            logvar (Tensor): The logarithm of the variance of the latent space distribution.

        Returns:
            Tensor: A sample from the latent distribution defined by mu and logvar.
        """
        std = torch.exp(0.5 * logvar)  # Calculate the standard deviation from log variance
        eps = torch.randn_like(std)  # Sample random noise having the same dimension as the standard deviation
        return mu + eps * std  # Return the reparameterized latent variable

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
    print(f'====> Test set average loss: {average_loss:.4f}')
    return average_loss

def main(junction_tensor, intron_tensor):
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
    train_loader, test_loader = setup_data_loaders(junction_tensor, intron_tensor, BATCH_SIZE, USE_CUDA)

    # Model Initialization
    model = VAE(INPUT_DIM, HIDDEN_DIMS, Z_DIM, OUTPUT_DIM)
    if USE_CUDA:
        model.cuda()

    # Optimizer Setup
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:  # Evaluate every 5 epochs and on the last epoch
            evaluate(model, test_loader)

if __name__ == "__main__":
    main()