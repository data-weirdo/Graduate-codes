import numpy as np
import torch
import pickle
from itertools import compress
from sklearn.metrics import roc_auc_score, average_precision_score


def Evaluation(dataloader, model, device):

    with open('./test_index.pickle', 'rb') as f:
        for_macro = pickle.load(f)
    f.close()

    y_labels_macro, y_scores_macro = [], []
    y_labels_micro, y_scores_micro = [], []
    for_macro = torch.tensor(for_macro)
    
    for _, (feature, label) in enumerate(dataloader):

        feature = feature.to(device)
        with torch.no_grad():

            outputs = model(feature)

        label = label.cpu()
        outputs = outputs.cpu()

        label_macro  = torch.index_select(label, dim=-1, index = for_macro)
        outputs_macro = torch.index_select(outputs, dim=-1, index = for_macro)

        y_labels_macro += label_macro.tolist()
        y_scores_macro += outputs_macro.tolist()
        y_labels_micro += label.tolist()
        y_scores_micro += outputs.tolist()

    y_labels_macro = np.array(y_labels_macro)
    y_scores_macro = np.array(y_scores_macro)
    y_labels_micro = np.array(y_labels_micro)
    y_scores_micro = np.array(y_scores_micro)

    auroc_macro = roc_auc_score(y_labels_macro, y_scores_macro)
    auroc_micro = roc_auc_score(y_labels_micro, y_scores_micro, average='micro')
    auprc_macro = average_precision_score(y_labels_macro, y_scores_macro)
    auprc_micro = average_precision_score(y_labels_micro, y_scores_micro, average='micro')

    return auroc_macro, auroc_micro, auprc_macro, auprc_micro