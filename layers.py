import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign  # RoIAlign module
from roi_align.roi_align import CropAndResize  # crop_and_resize module


class GCN(nn.Module):
    def __init__(self, cfg):
        super(GCN, self).__init__()
        self.cfg = cfg

        NFR = cfg.num_features_relation

        NG = cfg.num_graph
        N = cfg.num_boxes
        T = cfg.num_frames

        NFG = cfg.num_features_gcn
        NFG_ONE = NFG

        self.W = nn.Parameter(torch.zeros(size=(cfg.num_features_boxes, NFG)))
        self.fc_rn_theta = nn.Linear(NFG, NFR)
        self.fc_rn_phi = nn.Linear(NFG, NFR)

        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])

        self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([T * N, NFG_ONE]) for i in range(NG)])

    def forward(self, graph_boxes_features, boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, T, N, NFG = graph_boxes_features.shape
        NFR = self.cfg.num_features_relation
        NG = self.cfg.num_graph
        NFG_ONE = NFG
        OH, OW = self.cfg.out_size
        pos_threshold = self.cfg.pos_threshold

        # Prepare position mask
        graph_boxes_positions = boxes_in_flat  # B*T*N, 4
        graph_boxes_positions[:, 0] = (graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
        graph_boxes_positions[:, 1] = (graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
        graph_boxes_positions = graph_boxes_positions[:, :2].reshape(B*T, N, 2)  # B*T, N, 2

        # dot-product: sqrt(d)
        graph_boxes_distances = calc_pairwise_distance_3d(graph_boxes_positions, graph_boxes_positions)  # B, N, N

        # fs
        position_mask = (graph_boxes_distances > (pos_threshold * OW))
        position_mask = position_mask.reshape((B, T, N, N))

        graph_boxes_features_theta = self.fc_rn_theta(graph_boxes_features)  # B,N,NFR
        graph_boxes_features_phi = self.fc_rn_phi(graph_boxes_features)  # B,N,NFR

        similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                 graph_boxes_features_phi.transpose(2, 3))  # B,N,N

        similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)

        similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*N*N, 1

        # Build relation graph
        relation_graph = similarity_relation_graph

        relation_graph = relation_graph.reshape(B, T, N, N)

        relation_graph[position_mask] = -float('inf')

        relation_graph = torch.softmax(relation_graph, dim=3)

        # Graph convolution
        n2npool = torch.matmul(relation_graph, graph_boxes_features)
        graph_boxes_features = torch.matmul(n2npool, self.W)

        return graph_boxes_features, relation_graph


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_, adj):
        h = torch.matmul(input_, self.W)
        E, L, N, _ = h.size()

        a_input = torch.cat([h.repeat(1, 1, 1, N).view(E, L, N * N, -1), h.repeat(1, 1, N, 1)], dim=1).\
            view(E, L, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(4))

        attention = F.softmax(e, dim=3)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

