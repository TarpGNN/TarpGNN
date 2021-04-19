import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign  # RoIAlign module
from roi_align.roi_align import CropAndResize  # crop_and_resize module
from layers import GCN, GraphAttentionLayer


class GCNnet(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(GCNnet, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        self.inter = NFB + cfg.n_hid * cfg.n_heads
        # self.inter = NFG+NFB
        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=False)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)

        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.gcn_list = torch.nn.ModuleList([GCN(self.cfg) for i in range(self.cfg.gcn_layers)])

        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        self.fc_activities = nn.Linear(NFG, self.cfg.num_activities)

        self.attentions = [GraphAttentionLayer(NFB, cfg.n_hid, dropout=cfg.dropout, alpha=cfg.alpha, concat=True)
                           for _ in range(cfg.n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.attention_loc = nn.Sequential(
            nn.Linear(N, 16),
            nn.Sigmoid(),
            nn.Linear(16, N),
            nn.Sigmoid()
        )

        self.attention_interlayer = nn.Sequential(
            nn.Linear(self.inter, D),
            nn.Tanh(),
            nn.Linear(D, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.inter, self.inter),
            nn.ReLU(),
            nn.Linear(self.inter, cfg.num_activities),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, location_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # GCN
        graph_boxes_features = boxes_features
        relation_graph = None
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, boxes_in_flat)

        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, NFG)
        # graph_boxes_features.shape = [B, T, N, n_heads * n_hid]
        graph_boxes_features = torch.cat([att(graph_boxes_features, relation_graph) for att in self.attentions], dim=3)
        boxes_features = boxes_features.reshape(B, T, N, NFB)

        boxes_states = torch.cat([graph_boxes_features, boxes_features], dim=3)  # B, T, N, NFG+NFB
        node_loc = torch.unsqueeze(self.attention_loc(location_in), dim=2)
        node_loc = torch.transpose(node_loc, 3, 2)
        node_loc = node_loc.expand_as(boxes_states)
        boxes_states = torch.mul(boxes_states, node_loc)
        boxes_states = self.dropout_global(boxes_states)

        # Predict activities
        boxes_states = torch.cumsum(boxes_states, dim=2)[:, :, -1, :]
        boxes_states = boxes_states.div(N)
        A = self.attention_interlayer(boxes_states)
        A = torch.transpose(A, 2, 1)
        A = F.softmax(A, dim=2)
        boxes_states = torch.matmul(A, boxes_states)
        activities_scores = self.classifier(boxes_states.squeeze(1))

        return activities_scores
