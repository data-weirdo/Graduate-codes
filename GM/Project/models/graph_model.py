import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from models import register_model

@register_model("gat")
class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.num_layers = args.g_num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = args.g_activation
        self.g_heads = ([args.g_num_heads] * self.num_layers) + [1]

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            args.embedding_dim, args.g_hidden_dim, self.g_heads[0],
            args.g_feat_drop, args.g_attn_drop, args.g_negative_slope, 
            False, self.activation, allow_zero_in_degree=True))

        # hidden layers
        for l in range(1, self.num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                args.g_hidden_dim * self.g_heads[l-1], args.g_hidden_dim, 
                self.g_heads[l], args.g_feat_drop, args.g_attn_drop, 
                args.g_negative_slope, args.g_residual, \
                self.activation, allow_zero_in_degree=True))

        last_layers = GATConv(args.g_hidden_dim * self.g_heads[-2], args.g_hidden_dim, \
                    self.g_heads[-1], args.g_feat_drop, args.g_attn_drop, args.g_negative_slope, 
                    args.g_residual, None, allow_zero_in_degree=True)

        # output projection
        self.gat_layers.append(last_layers)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, g):
        feat = g.ndata['x']
        for l in range(self.num_layers):
            feat = self.gat_layers[l](g, feat).flatten(1)
        # output projection
        net_output = self.gat_layers[-1](g, feat).mean(1) 
        
        g_clone = g.clone()
        g_clone.ndata['x'] = net_output

        net_output = dgl.unbatch(g_clone)

        return net_output
