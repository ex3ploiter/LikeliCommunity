from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Data
from GC_SEE_module.GCN import GCN


class Encoder(nn.Module):
    def __init__(self,  hidden_dim, input_dim):
        super().__init__()

        self.hidden_model = nn.ModuleList([
            GCN(input_dim, hidden_dim),
            GCN(hidden_dim, hidden_dim),
            GCN(hidden_dim, hidden_dim)
        ])

        self.mean_model = nn.ModuleList([
            GCN(hidden_dim, hidden_dim),
            GCN(hidden_dim, hidden_dim)
        ])

        self.std_model = nn.ModuleList([
            GCN(hidden_dim, hidden_dim),
            GCN(hidden_dim, hidden_dim)
        ])

    def encode(self,
               x: torch.Tensor,
               edge_index: torch.LongTensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:

        # hidden = self.hidden_model(x, edge_index)
        hidden = x
        for layer in self.hidden_model:
            hidden = layer(hidden, edge_index)

        # mean = self.mean_model(hidden, edge_index)
        mean = hidden
        for layer in self.mean_model:
            mean = layer(mean, edge_index)

        # std = self.std_model(hidden, edge_index)
        std = hidden
        for layer in self.std_model:
            std = layer(std, edge_index)

        return mean, std

    def forward(self, X, adj):
        x, edge_index = X, adj
        mu, logvar = self.encode(x, edge_index)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout: float = 0.1):
        super().__init__()

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dropout(z)
        adj_reconstructed = torch.matmul(z, z.T)

        if self.training:
            adj_reconstructed = self.activation(adj_reconstructed)

        return adj_reconstructed


class VGAE_New(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()

        self.encoder = Encoder(hidden_dim, input_dim)
        self.decoder = Decoder()

    def reparametrize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, X, adj_):
        mu, logvar = self.encoder(X, adj_)
        z = self.reparametrize(mu, logvar)
        adj = self.decoder(z)
        return adj, mu, mu
