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
from VGAE.VGAE_loss import VGAELoss


class LikelihoodComputer(nn.Module):
    def __init__(self, dataset):
        super(LikelihoodComputer, self).__init__()
        self.config = config = {
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
            "LR": 0.001,
            "EPOCHS": 10 , # Adjust the number of epochs as needed
            "hidden_dim" : 64
        }
        self.device = self.config["DEVICE"]
        self.hidden_dim = self.config["hidden_dim"]  # Adjust as needed
        self.dataset = dataset

        # Create models

        self.hidden_model, self.mean_model, self.std_model = get_VGAE_hidden_models(
            dataset, self.hidden_dim)

        # Initialize encoder and decoder
        self.encoder = Encoder(
            hidden_model=self.hidden_model,
            mean_model=self.mean_model,
            std_model=self.std_model
        )

        self.decoder = Decoder()

        # Create VGAE model
        self.model = VGAE(
            encoder=self.encoder,
            decoder=self.decoder
        )

        # Define loss function and optimizer
        self.loss_function = VGAELoss(norm=2)
        self.optimizer = AdamW(params=self.model.parameters(), lr=config["LR"])

        self.train()

    def train_epoch(self, data):
        self.model.train()
        self.model.to(self.device)
        self.loss_function.to(self.device)

        preds = []
        targets = []
        total_loss = 0.

        adj = adj_matrix_from_edge_index(data.x, data.edge_index)

        self.optimizer.zero_grad()
        adj_output, mu, logvar = self.model.to(self.device)(data.to(self.device))

        loss = self.loss_function(adj_output, mu, logvar, adj.to(self.device))

        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # preds.append(adj_output.flatten())
        # targets.append(adj.flatten())

        # preds = torch.cat(preds, dim=0).sigmoid().detach().cpu().numpy()
        # targets = torch.cat(targets, dim=0).detach().cpu().numpy()
        # roc_auc = roc_auc_score(targets, preds)

        # print(f"TRAIN Loss: {total_loss}, ROC AUC: {roc_auc}")
        
        # print(f"VGAE TRAIN Loss: {total_loss}")

    def train(self):
        for epoch in range(self.config["EPOCHS"]):
            self.train_epoch(
                data=self.dataset,
            )

    def ComputeLikelihood(self):
        adj_output, _, _ = self.model(self.dataset)
        adj_output = nn.Sigmoid()(adj_output)
        return adj_output.sum()

    def forward(self):
        return self.ComputeLikelihood()
