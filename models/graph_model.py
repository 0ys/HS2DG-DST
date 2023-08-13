"""
graph_model.py: GAT model for the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_modules import GraphFilterBatchAttentional

# "graph_model": {
#     "model_type": "GAT",
#     "num_layer": 4,
#     "num_head": 4,
#     "feature_size": 768,
#     "num_hop": 2,
#     "graph_dropout": 0.2,
# },

class GraphModel(nn.Module):
    """
        Graph Models used in paper
    """
    def __init__(self, args):
        super(GraphModel,self).__init__()
        self.args = args
        self.graph_layers = nn.ModuleList()
        num_layer = self.args.num_layer
        G = self.args.feature_size
        F = self.args.feature_size
        P = self.args.num_head
        K = self.args.num_hop

        for _ in range(num_layer):
            self.graph_layers.append(
                GraphFilterBatchAttentional(G=G, 
                                            F=F, 
                                            K=K, 
                                            P=P, 
                                            concatenate=False, 
                                            bias=False)
            )
        self.dropout = nn.Dropout(self.args.graph_dropout)

    def add_GSO(self, S):
        """Add GSO (ontology descriptor)

        Args:
            S (Tensor): ontology descriptor B x E x N x N
        """
        for graph_layer in self.graph_layers:
            graph_layer.addGSO(S)

    def get_GSO(self):
        attentions = []
        for graph_layer in self.graph_layers:
            attentions += [graph_layer.returnAttentionGSO()]
        return attentions

    def forward(self, x):
        y = x
        for graph_layer in self.graph_layers:
            y = graph_layer(y)
        y = self.dropout(y)
        return y
