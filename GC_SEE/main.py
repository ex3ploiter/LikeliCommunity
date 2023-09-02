import torch
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from sklearn.cluster import KMeans

from GC_SEE.GC_SEE_model.GCSEE.model import GCSEE
from GC_SEE.GC_SEE_utils import data_processor
from GC_SEE.GC_SEE_utils.evaluation import eva
from GC_SEE.GC_SEE_utils.result import Result
from GC_SEE.GC_SEE_utils.utils import count_parameters, get_format_variables

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

# from modelX import VGAE, Encoder, Decoder
from VGAE.VGAE_loss import VGAELoss
# from utils import adj_matrix_from_edge_index