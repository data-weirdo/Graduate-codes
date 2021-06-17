import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.join(dirname, 'utils')
sys.path.append(dirname)

from .models import *
from utils import *
# from torchvision import models

start_time = time.time()
def trainer(args, train_loader, train_len, valid_loader, valid_len, device, wandb, pos_weight):
    model = ResNet50(args.dropout_rate).to(device)
    # model = models.resnet50(pretrained=True).to(device)
    model = nn.DataParallel(model, device_ids = [0,1,2,3,4,5,6,7])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 10 * len(train_loader) , gamma=0.1)
    # pos_weight = pos_weight.to(device)
    # BCELoss = nn.BCELoss(weight=pos_weight)
    BCELoss = nn.BCELoss()
    label_cnt = 0
    
    for epoch in range(args.n_epochs):
        train_loss = 0
        train_correct = 0

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.float().to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = BCELoss(logits, targets)
            predicted_labels = torch.where(logits>0.5, 1, 0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_correct += (predicted_labels == targets).sum()

            if batch_idx == 0:
                label_cnt = targets.size(1)

            if batch_idx % 50 == 0:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                    %(epoch+1, args.n_epochs, batch_idx, len(train_loader), loss))

        model.eval()
        with torch.no_grad():
            valid_info = ValidEval(model, valid_loader, valid_len, device, BCELoss)
            print('Epoch: %03d/%03d | Train Accuracy: %.3f%% | Valid Accuracy: %.3f%%' %(
                epoch+1, args.n_epochs, train_correct.float()/train_len/label_cnt*100, valid_info[1]/label_cnt
            ))

            # As a result of several experiments, I concluded that if I can get a model whose validation accuracy is over 88.7, 
            # maybe I can get a score outperforming the TA's.
            if (valid_info[1] / label_cnt) > 88.9:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), './model_param_temp.pt')
                else:
                    torch.save(model.state_dict(), './model_param_temp.pt')
                return model
                
        wandb.log({"Train Loss": train_loss / len(train_loader),
                "Valid Loss": valid_info[0],
                "Train Accuracy": train_correct / train_len / label_cnt * 100,
                "Validation Accuracy": valid_info[1] / label_cnt})

    end_time = time.time()
    print('Total training time: %.2f min' %((end_time - start_time)/60))

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), args.model_dir)
    else:
        torch.save(model.state_dict(), args.model_dir)

    return model

def inference(args, model, test_loader, device):
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(args.model_dir))
    else:
        model.load_state_dict(torch.load(args.model_dir))

    with torch.set_grad_enabled(False):
        model.to(device)
        auroc_macro, auroc_micro, auprc_macro, auprc_micro = InferenceEval(test_loader, model, device)

        text_to_record = '20213207' + '\n' + \
            str(auroc_macro) + '\n' + str(auroc_micro) + '\n' + \
            str(auprc_macro) + '\n' + str(auprc_micro)

        with open(args.out_dir, 'w') as f:
            f.write(text_to_record)
        f.close()

    return None
