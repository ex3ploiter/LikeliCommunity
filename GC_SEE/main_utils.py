import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import DataLoader

def get_VGAE_hidden_models(dataset, hidden_dim=64):
    hidden_model = Sequential('x, edge_index, edge_attr', [
        (GCNConv(dataset.num_features, hidden_dim), 'x, edge_index -> x1'),
        nn.ReLU(inplace=True),
        (GCNConv(hidden_dim, hidden_dim), 'x1, edge_index -> x2'),
        nn.ReLU(inplace=True),
        (GCNConv(hidden_dim, hidden_dim), 'x2, edge_index -> x3'),
        nn.ReLU(inplace=True),
    ])

    mean_model = Sequential('x, edge_index, edge_attr', [
        (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x1'),
        nn.ReLU(inplace=True),
        (GCNConv(hidden_dim, hidden_dim), 'x1, edge_index -> x2'),
        nn.ReLU(inplace=True),
    ])

    std_model = Sequential('x, edge_index, edge_attr', [
        (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x1'),
        nn.ReLU(inplace=True),
        (GCNConv(hidden_dim, hidden_dim), 'x1, edge_index -> x2'),
        nn.ReLU(inplace=True),
    ])

    return hidden_model, mean_model, std_model
