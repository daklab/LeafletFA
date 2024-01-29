# factor_model.py 
# maintainer: Karin Isaev 
# date: 2024-01-26

# purpose:  
#   - define VAE model with pyro using just splicing information
#   - framework based on tutorial from https://pyro.ai/examples/vae.html

import os

import numpy as np
import torch
#from pyro.contrib.examples.util import MNIST
import torch.nn as nn
#import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
#import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

#assert pyro.__version__.startswith('1.8.6')
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

import random
import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

class CustomDataset(Dataset):
    def __init__(self, junction_tensor, cluster_tensor):
        self.junction_tensor = junction_tensor.to_dense()  # Convert to dense if needed
        self.cluster_tensor = cluster_tensor.to_dense()    # Renamed for clarity

    def __len__(self):
        return self.junction_tensor.shape[0]

    def __getitem__(self, idx):
        junction_data = self.junction_tensor[idx]
        cluster_data = self.cluster_tensor[idx]
        return junction_data, cluster_data


def setup_data_loaders(junction_tensor, cluster_tensor, batch_size=128, use_cuda=False):
    
    # if using cuda move junction tesnor and cluster tensor to cuda
    if use_cuda:
        junction_tensor = junction_tensor.cuda()
        cluster_tensor = cluster_tensor.cuda()
    dataset = CustomDataset(junction_tensor, cluster_tensor)
    
    # Determine the generator device type based on whether CUDA is used
    generator = torch.Generator(device="cuda" if use_cuda else "cpu")

    # Splitting the dataset into train and test
    train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
    test_size = len(dataset) - train_size  # Remaining 20% for testing

    # Make sure random-split is on cpu or else it will be very slow
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 2)  # Outputs two parameters for the beta distribution
        self.softplus = nn.Softplus()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        beta_params = self.softplus(self.fc21(hidden))
        return beta_params

#  Package the model and guide in a PyTorch module: 
class VAE(nn.Module):
    def __init__(self, input_dim, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(input_dim, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
        self.z_dim = z_dim

    def model(self, junction_data, cluster_data):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", junction_data.shape[0]):
            z_loc = junction_data.new_zeros(torch.Size((junction_data.shape[0], self.z_dim)))
            z_scale = junction_data.new_ones(torch.Size((junction_data.shape[0], self.z_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            beta_params = self.decoder(z)
            alpha, beta = beta_params[:, 0], beta_params[:, 1]

            # Sample from beta distribution for PSI
            pyro.sample("psi", dist.Beta(alpha, beta).to_event(1), obs=self.compute_psi(junction_data, cluster_data))

            # Sample from a binomial distribution for junction counts
            # Here, you need to define or calculate 'total_count' and 'probs'
            pyro.sample("junction", dist.Binomial(total_count=..., probs=...), obs=junction_data)

    def guide(self, junction_data, cluster_data):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", junction_data.shape[0]):
            z_loc, z_scale = self.encoder(junction_data)  # Assuming junction_data is used for encoding
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    @staticmethod
    def compute_psi(junction_data, cluster_data):
        psi_values = junction_data / torch.clamp(cluster_data, min=1)
        return psi_values

def train(svi, train_loader, use_cuda=False):
    epoch_loss = 0.
    for junction_data, cluster_data in train_loader:
        if use_cuda:
            junction_data = junction_data.cuda()
            cluster_data = cluster_data.cuda()
        epoch_loss += svi.step(junction_data, cluster_data)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    test_loss = 0.
    for junction_data, cluster_data in test_loader:
        if use_cuda:
            junction_data = junction_data.cuda()
            cluster_data = cluster_data.cuda()
        test_loss += svi.evaluate_loss(junction_data, cluster_data)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

def main(junction_tensor, cluster_tensor):
    # Run options
    LEARNING_RATE = 1.0e-3
    USE_CUDA = torch.cuda.is_available()

    # Run only for a single iteration for testing
    NUM_EPOCHS = 100
    TEST_FREQUENCY = 5

    train_loader, test_loader = setup_data_loaders(junction_tensor, cluster_tensor, batch_size=256, use_cuda=USE_CUDA)

    # Clear param store
    pyro.clear_param_store()

    # Setup the VAE
    input_dim = junction_tensor.shape[1] # check dimension 
    vae = VAE(input_dim, use_cuda=USE_CUDA)

    # Setup the optimizer
    adam_args = {"lr": LEARNING_RATE}
    optimizer = Adam(adam_args)

    # Setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % TEST_FREQUENCY == 0:
            # Report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

if __name__ == "__main__":
    main()

