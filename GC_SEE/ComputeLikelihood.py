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
from VGAE.VGAE_loss import VGAELoss,VGAELoss_Main


class LikelihoodComputer(nn.Module):
    def __init__(self, feature,adj,model):
        super(LikelihoodComputer, self).__init__()
        self.config = config = {
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
            "LR": 0.01,
            "EPOCHS": 10 , # Adjust the number of epochs as needed
            "hidden_dim" : 128
        }
        self.device = self.config["DEVICE"]
        self.hidden_dim = self.config["hidden_dim"]  # Adjust as needed
        self.feature=feature
        self.adj=adj

        # Create models

 

        # Initialize encoder and decoder
      

        # Create VGAE model
        # self.model = VGAE()
        # self.model = VGAE_Model()
        self.model=model

        # Define loss function and optimizer
        self.loss_function = VGAELoss_Main(norm=2)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=config["LR"])

        self.train()

    def train_epoch(self):
        self.model.train()
        self.model.to(self.device)
        self.loss_function.to(self.device)

        preds = []
        targets = []
        total_loss = 0.

        # adj = adj_matrix_from_edge_index(self.feature, self.adj)
        adj=self.adj

        self.optimizer.zero_grad()
        adj_output, mu, logvar = self.model.to(self.device)(self.feature,self.adj)

        loss = self.loss_function(adj_output, mu, logvar, adj.to(self.device))

        total_loss += loss.item()

        loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        # print(f"VGAE TRAIN Loss: {total_loss}")

    def train(self):
        for epoch in range(self.config["EPOCHS"]):
            self.train_epoch()

    def ComputeLikelihood(self):
        adj_output, _, _ = self.model(self.feature,self.adj)
        adj_output = nn.Sigmoid()(adj_output)
        return -adj_output.mean()

    def forward(self):
        return self.ComputeLikelihood()