# ============================================================================
# MODELOS GNN — GCN, GraphSAGE, GAT (PyTorch Geometric)
# Arquitetura aprimorada:
#   - 3 camadas de convolução (agrega vizinhança de até 3 hops)
#   - Batch Normalization entre camadas (estabiliza treino)
#   - Residual connections (evita over-smoothing em grafos esparsos)
#   - Jumping Knowledge (JK-concat): concatena saídas de todas as camadas
#     antes da classificação, preservando informação local e global
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, JumpingKnowledge


class GCN(torch.nn.Module):
    """GCN com 3 camadas, BatchNorm, residual connections e Jumping Knowledge."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5, num_layers: int = 3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Projeção inicial para hidden_channels
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # JK-concat: concatena saídas de todas as camadas
        self.jk = JumpingKnowledge(mode='cat')
        self.classifier = nn.Linear(hidden_channels * num_layers, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_proj(x))

        layer_outs = []
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = x + residual          # residual connection
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outs.append(x)

        x = self.jk(layer_outs)       # concat das 3 camadas
        return self.classifier(x)


class GraphSAGE(torch.nn.Module):
    """GraphSAGE com 3 camadas, BatchNorm, residual e Jumping Knowledge."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5, num_layers: int = 3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.jk = JumpingKnowledge(mode='cat')
        self.classifier = nn.Linear(hidden_channels * num_layers, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_proj(x))

        layer_outs = []
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = x + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outs.append(x)

        x = self.jk(layer_outs)
        return self.classifier(x)


class GAT(torch.nn.Module):
    """GAT com 3 camadas, BatchNorm, residual e Jumping Knowledge.
    Atenção multi-head em todas as camadas (não só na primeira)."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 heads: int = 4, dropout: float = 0.5, num_layers: int = 3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Projeção para hidden_channels (tamanho fixo entre camadas)
        self.input_proj = nn.Linear(in_channels, hidden_channels * heads)

        for _ in range(num_layers):
            # concat=False: média dos heads → hidden_channels (não explode em cada camada)
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                       heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.jk = JumpingKnowledge(mode='cat')
        self.classifier = nn.Linear(hidden_channels * heads * num_layers, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.input_proj(x))

        layer_outs = []
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = x + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outs.append(x)

        x = self.jk(layer_outs)
        return self.classifier(x)