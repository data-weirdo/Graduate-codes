import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt


def compute_stats(model, data_loader, data_len, device, loss_func, argument):
    correct_pred= 0 
    valid_loss = 0

    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)

        if loss_func != None: # None: Train data case
            loss = loss_func(logits, targets) 

            if argument.reg in [2,3]:
                if argument.reg == 2: # L2 norm
                    reg = 1e-2 * torch.sqrt(sum((p**2).sum() for p in model.parameters()))
                else: # L1 norm
                    reg = 1e-5 * sum(p.abs().sum() for p in model.parameters())
                
                loss += reg
            
            valid_loss += loss

        _, predicted_labels = torch.max(probas, 1)
        correct_pred += (predicted_labels == targets).sum()

    if loss_func != None:
        valid_loss = valid_loss / len(data_loader) # don't need in test case
    else:
        valid_loss = None

    accuracy = correct_pred.float()/data_len * 100
    accuracy = accuracy.item()

    return valid_loss, accuracy
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def return_name(args_model, args_reg, args_optim):

    if args_model == 1:
        model = 'MLP'
    elif args_model == 2:
        model = 'VGG19'
    else:
        model = 'ResNet-18'


    if args_reg == 0:
        reg = 'No_Regularization'
    elif args_reg == 1:
        reg = 'Dropout'
    elif args_reg == 2:
        reg = 'L2_Norm'
    else:
        reg = 'L1_Norm'


    if args_optim == 1:
        optim = 'SGD_with_momentum'
    elif args_optim == 2:
        optim = 'AdaGrad'
    else:
        optim = 'Adam'

    return model, reg, optim 