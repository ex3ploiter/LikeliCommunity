import torch
import torch.nn as nn
# from torch_geometric.nn import GCN, Sequential
from torch_geometric.data import DataLoader
from GC_SEE_module.GCN import GCN

def get_VGAE_hidden_models(X, hidden_dim=64):
    hidden_model = nn.Sequential(
        GCN(X.shape[1], hidden_dim),
        GCN(hidden_dim, hidden_dim),
        GCN(hidden_dim, hidden_dim),
    )

    mean_model = nn.Sequential(
        GCN(hidden_dim, hidden_dim),
        GCN(hidden_dim, hidden_dim),
    )

    std_model = nn.Sequential(
        GCN(hidden_dim, hidden_dim),
        GCN(hidden_dim, hidden_dim),
        
    )

    return hidden_model, mean_model, std_model
