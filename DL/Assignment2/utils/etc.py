import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.join(dirname, 'src')
sys.path.append(dirname)

from trainer import *
from model import LSTMModel, TransformerModel, BERTModel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def wandb_logging_name(model_args):
    assert model_args in [1, 2, 3], 'Wrong model args'

    if model_args == 1:
        model_name = 'LSTM'
    elif model_args == 2:
        model_name = 'Transformer'
    else:
        model_name = 'BERT-mini'

    return model_name


def model_selection(model_args, model_checkpoint=None, dim=None, num_heads=None, num_layers=None):
    assert model_args in [1, 2, 3], 'Wrong model args'

    if model_args == 1:
        model = LSTMModel(dim)
    elif model_args == 2:
        model = TransformerModel(dim, num_heads, num_layers)
    else:
        model = BERTModel.from_pretrained(model_checkpoint)

    return model
