import os
import logging
import sys
import torch
import torch.nn as nn

from copy import deepcopy
from models import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.join(dirname, 'utils')
sys.path.append(dirname)

from utils.model_utils import *

@register_model("pred_model")
class PredModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.online_encoder = self._gnn_model.build_model(args)
        self.target_encoder = deepcopy(self.online_encoder)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.target_ema_updater = ExpMovingAvg(args.ma_decay, args.epochs)

        self.online_predictor = nn.Sequential(nn.Linear(args.g_hidden_dim, args.g_hidden_dim*2), 
                                            nn.PReLU(), 
                                            nn.Linear(args.g_hidden_dim*2, args.g_hidden_dim))

        self.supervised_predictor = nn.Linear(args.g_hidden_dim*2, 2)

    @property
    def _gnn_model(self):
        if self.args.graph_model is not None:
            return MODEL_REGISTRY[self.args.graph_model]
        else:
            return None

    @classmethod
    def build_model(cls, args):
        return cls(args)

    def update_moving_average(self):
        ma_update(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, aug1, aug2, edge_pair, device, mode):
        if mode == 'train':
            aug1_online = self.online_encoder(aug1)[0]
            aug2_online = self.online_encoder(aug2)[0]

            aug1_pred = self.online_predictor(aug1_online.ndata['x'])
            aug2_pred = self.online_predictor(aug2_online.ndata['x'])
            
            with torch.no_grad():
                v1_target = self.target_encoder(aug1)[0].ndata['x']
                v2_target = self.target_encoder(aug2)[0].ndata['x']
                
            loss1 = byol_loss(aug1_pred, v2_target.detach())
            loss2 = byol_loss(aug2_pred, v1_target.detach())
            loss = loss1 + loss2

            aug1_concatenated = get_paired_embedding(aug1_online.ndata['x'], edge_pair.to(device))
            supervised_logit = self.supervised_predictor(aug1_concatenated)

            return loss.mean(), supervised_logit

        else:
            aug1_online = self.online_encoder(aug1)[0]
            aug1_concatenated = get_paired_embedding(aug1_online.ndata['x'], edge_pair.to(device))
            supervised_logit = self.supervised_predictor(aug1_concatenated)

            return supervised_logit