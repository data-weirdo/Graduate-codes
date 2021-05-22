import sys
import os
import time
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.join(dirname, 'utils')
sys.path.append(dirname)

from .models import *
from utils import *

start_time = time.time()
def trainer(args, train_loader, train_len, valid_loader, valid_len, device, wandb):
    model = model_selection(args.model, 
                            args.reg, 
                            args.drop_param).to(device)
    optimizer = optimizer_selection(model, args.optimizer, args.lr, args.momentum)

    CELoss = nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):
        train_loss = 0
        train_correct = 0

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            logits, probs = model(features)
            loss = CELoss(logits, targets)

            if args.reg in [2,3]:
                if args.reg == 2: # L2 norm
                    reg = 5e-3 * torch.sqrt(sum((p**2).sum() for p in model.parameters()))

                else: # L1 norm
                    reg = 5e-6 * sum(p.abs().sum() for p in model.parameters())
                
                loss += reg

            _, predicted_labels = torch.max(probs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (predicted_labels == targets).sum()

            if batch_idx % 50 == 0:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                    %(epoch+1, args.n_epochs, batch_idx, len(train_loader), loss))

        model.eval()
        with torch.no_grad():
            valid_info = compute_stats(model, valid_loader, valid_len, device, CELoss, args)
            print('Epoch: %03d/%03d | Train Accuracy: %.3f%% | Valid Accuracy: %.3f%%' %(
                epoch+1, args.n_epochs, train_correct.float()/train_len*100, valid_info[1]
            ))
                
        wandb.log({"Train Loss": train_loss / len(train_loader),
                "Valid Loss": valid_info[0],
                "Train Accuracy": train_correct / train_len * 100,
                "Validation Accuracy": valid_info[1]})

    end_time = time.time()
    print('Total training time: %.2f min' %((end_time - start_time)/60))

    return model

def inference(args, model, test_loader, test_len, device, wandb):
    model.eval()
    with torch.set_grad_enabled(False):
        print('Test Accuracy: %.3f%%' %(compute_stats(model, test_loader, test_len, device, loss_func=None, argument=args)[1]))    
        wandb.log({'Test Accuracy': compute_stats(model, test_loader, test_len, device, loss_func=None, argument=args)[1]})

    return None

# def show_image(args, model, test_loader, device, wandb):

#     bsz = args.batch_size
#     last = len(test_loader) // bsz
#     model.eval()
#     with torch.set_grad_enabled(False):
#         for batch_idx, (features, targets) in enumerate(test_loader):
#             if batch_idx == last-1:
#                 features = features.to(device)
#                 print('Print one of the pictures')
#                 plt.imshow(np.transpose(features[0], (1,2,0)))
#                 wandb.log({'Example test image': wandb.Image(plt.imshow(np.transpose(features[0], (1,2,0))))})

#     return None