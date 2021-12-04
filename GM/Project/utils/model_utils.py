import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl


def get_paired_embedding(embedding, edge_pair):
    assert edge_pair.size(0) % 2 == 0, 'For convenience, make your batch size as even number'
    row_cnt = edge_pair.size(0)
    edge_pair_align = edge_pair.view(-1)
    emb_concat = torch.index_select(embedding, 0, edge_pair_align)
    emb_concat = emb_concat.view(row_cnt, -1)
    return emb_concat

class Augmentation:
    def __init__(self, 
            feat_aug1_p = 0.5, 
            feat_aug2_p = 0.5,
            edge_aug1_p = 0.5,
            edge_aug2_p = 0.5
            ):

        self.feat_aug1_p = feat_aug1_p
        self.feat_aug2_p = feat_aug2_p
        self.edge_aug1_p = edge_aug1_p
        self.edge_aug2_p = edge_aug2_p
    
    def _augmentation_with_masking(self, g, device):
        # node feature masking
        node_size, feat_dim = g.ndata['x'].size()
        feat_mask1 = torch.FloatTensor(node_size, feat_dim).uniform_() > self.feat_aug1_p
        feat_mask2 = torch.FloatTensor(node_size, feat_dim).uniform_() > self.feat_aug2_p
        original_feat1, original_feat2 = g.ndata['x'].clone(), g.ndata['x'].clone()
        aug_feat1, aug_feat2 = original_feat1 * feat_mask1.to(device), original_feat2 * feat_mask2.to(device)

        # edge connectivity masking 
        src_edge1, dst_edge1 = self._edge_drop(g.edges(), self.edge_aug1_p)
        src_edge2, dst_edge2 = self._edge_drop(g.edges(), self.edge_aug2_p)

        # update
        aug_g1 = dgl.graph((src_edge1, dst_edge1))
        aug_g2 = dgl.graph((src_edge2, dst_edge2))
        aug_g1.ndata['x'] = aug_feat1
        aug_g2.ndata['x'] = aug_feat2
        
        return aug_g1.to(device), aug_g2.to(device)

    def _edge_drop(self, edges, probs):
        src, dst = edges
        mask = torch.vstack(edges).new_full((src.size(0),), 1-probs, dtype=torch.float)
        mask = torch.bernoulli(mask).to(torch.bool)
        new_src, new_dst = self._filter_adj(src, dst, mask)
        return new_src, new_dst

    def _filter_adj(self, row, col, mask):
        return row[mask], col[mask]

    def __call__(self, g, device):
        return self._augmentation_with_masking(g, device)

class ExpMovingAvg:
    def __init__(self, tau, epochs):
        super().__init__()
        self.tau = tau
        self.k = 0
        self.K = epochs

    def ema_weighted_average(self, old, new):
        if old is None:
            return new
        tau = 1 - (1 - self.tau) * (np.cos(np.pi * self.k / self.K) + 1) / 2.0
        self.k += 1
        return tau * old + (1 - tau) * new

def byol_loss(q, z):
    q = F.normalize(q, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (q * z).sum(dim=-1)

def ma_update(ema_updater, target_enc, online_enc):
    for current_params, ma_params in zip(online_enc.parameters(), target_enc.parameters()):
        xi, theta = ma_params.data, current_params.data
        ma_params.data = ema_updater.ema_weighted_average(xi, theta)