import sys
import os
import torch
import torch.optim as optim

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.join(dirname, 'src')
sys.path.append(dirname)

from src.models import MLP,  VGG19, ResNet18

# Choose the model to use
def model_selection(model_args, reg_args, dp_args):
    """
    model_args: model number
    reg_args: regularization number
    dp_args: dropout rate
    """

    if model_args == 1:
        return MLP(reg_args, dp_args)

    elif model_args == 2:
        return VGG19(reg_args=reg_args, dropout_rate=dp_args)

    else:
        return ResNet18(reg_args=reg_args, dropout_rate=dp_args)
    

# Choose the optimizer to use
def optimizer_selection(model, optimizer_args, lr_args, momentum_args):
    if optimizer_args == 1:
        optimizer = optim.SGD(model.parameters(), lr=lr_args, momentum=momentum_args)

    elif optimizer_args == 2:
        optimizer = optim.Adagrad(model.parameters(), lr=lr_args)

    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_args)
    
    return optimizer
