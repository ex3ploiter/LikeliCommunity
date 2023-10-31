from VGAE.VGAE_model import VGAE, Encoder, Decoder
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Subset

from tqdm.notebook import tqdm

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from VGAE.VGAE_utils import adj_matrix_from_edge_index
from main_utils import get_VGAE_hidden_models
from VGAE.VGAE_loss import VGAELoss, VGAELoss_Main
from VGAE.VGAE_pyG import DeepVGAE


class LikelihoodComputer_pyG(nn.Module):
    def __init__(self, feature, adj):
        super(LikelihoodComputer_pyG, self).__init__()
        self.config = config = {
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
            "LR": 0.01,
            "EPOCHS": 10,  # Adjust the number of epochs as needed
            "hidden_dim": 128
        }
        self.device = self.config["DEVICE"]

        self.feature = feature
        self.edge = adj.nonzero().t().contiguous()
        self.adj = adj

        # Define loss function and optimizer
        self.model = DeepVGAE(
            enc_in_channels=self.feature.shape[1], enc_hidden_channels=64, enc_out_channels=32).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=config["LR"])

        self.train()

    def train_epoch(self):
        self.model.train()
        self.model.to(self.device)

        self.optimizer.zero_grad()

        loss = self.model.loss(self.feature, self.edge, self.adj)

        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()

    def train(self):
        # print("\n")
        for epoch in range(self.config["EPOCHS"]):
            loss = self.train_epoch()
            # print(f"VGAE TRAIN Loss: {loss}")
        # print("\n")

    def ComputeLikelihood(self):
        adj_output = self.model(self.feature, self.edge)
        adj_output = nn.Sigmoid()(adj_output)
        return -adj_output.mean()

    def forward(self):
        return self.ComputeLikelihood()
