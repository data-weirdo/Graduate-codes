import numpy as np
import torch
import pickle
from itertools import compress
from sklearn.metrics import roc_auc_score, average_precision_score


def ValidEval(model, dataloader, data_len, device, loss_func):
    valid_loss = 0
    valid_correct = 0

    for _, (feature, label) in enumerate(dataloader):
        feature = feature.float().to(device)
        label = label.to(device)
        
        with torch.no_grad():
            logit = model(feature)
            predicted_labels = torch.where(logit>0.5, 1, 0)
            loss = loss_func(logit, label)
            
            predicted_labels = predicted_labels.cpu()
            label = label.cpu()
            
            valid_loss += loss.item()
            valid_correct += (predicted_labels == label).sum()

    return valid_loss / len(dataloader) , valid_correct / data_len * 100


def InferenceEval(dataloader, model, device):
    y_labels, y_scores = [], []
    
    for _, (feature, label) in enumerate(dataloader):
        feature = feature.float().to(device)
        with torch.no_grad():
            outputs = model(feature)

        label = label.cpu()
        outputs = outputs.cpu()

        y_labels += label.tolist()
        y_scores += outputs.tolist()

    y_labels = np.array(y_labels)
    y_scores = np.array(y_scores)

    auroc_macro = roc_auc_score(y_labels, y_scores)
    auroc_micro = roc_auc_score(y_labels, y_scores, average='micro')
    auprc_macro = average_precision_score(y_labels, y_scores)
    auprc_micro = average_precision_score(y_labels, y_scores, average='micro')

    return auroc_macro, auroc_micro, auprc_macro, auprc_micro