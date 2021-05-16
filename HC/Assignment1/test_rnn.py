import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import TensorDataset, DataLoader, dataset, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRU(nn.Module):
    def __init__(self, item_len, value_len, embedding_dim, hidden_dim, last_dim, num_layers, seq_len=100):
        super(GRU, self).__init__()

        self.e_dim = embedding_dim
        self.h_dim = hidden_dim 
        self.last_dim = last_dim 
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        self.itemid_embedding = nn.Embedding(item_len, self.e_dim, padding_idx=0)
        self.valuenum_embedding = nn.Embedding(value_len, self.e_dim, padding_idx=0)
        self.time_embedding = nn.Embedding(seq_len, self.e_dim)
        
        self.gru = nn.GRU(self.e_dim*3, self.h_dim, num_layers=self.num_layers, batch_first=True)

        self.linear = nn.Linear(self.h_dim*self.num_layers, self.last_dim)
        self.relu = nn.ReLU()
        self.output_fc = nn.Linear(self.last_dim, 2)

    def forward(self, x):
        
        itemids = x['item_ids'] # (batch_size, seq_len)
        valuenums = x['value_nums'] # (batch_size, seq_len)
        timeids = x['time_ids'] # (batch_size, seq_len)
        lengths = x['lengths'] # (batch_size ,)
        
        i_emb = self.itemid_embedding(itemids)
        v_emb = self.valuenum_embedding(valuenums)
        t_emb = self.time_embedding(timeids)

        bsz = itemids.size(0)
        masks = torch.zeros(bsz, self.seq_len).float().cuda()

        for idx, length in enumerate(lengths):
            masks[idx][:length] = 1.0

        i_emb = i_emb * masks.unsqueeze(2)
        v_emb = v_emb * masks.unsqueeze(2)
        t_emb = t_emb * masks.unsqueeze(2)

        gru_inputs = torch.cat([i_emb, v_emb, t_emb], dim=2)
        packed_rnn_inputs = pack_padded_sequence(gru_inputs, lengths.tolist(), batch_first=True)
        
        _, hidden = self.gru(packed_rnn_inputs)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous().view(bsz, -1)
        outputs = self.output_fc(self.relu(self.linear(hidden)))
        
        return outputs

@dataclass
class DataCollatorWithSorting:
    def __call__(self, features):
        
        # instances = (batch, 3, embdding_dim)
        # labels = (batch)
        instances, labels = self._organize_batch(features)
        itemids, valuenums = instances[:, 0, :], instances[:, 1, :]
        timeids = instances[:, 2, :]
        
        lengths = (itemids > 0).sum(1)
        
        sorted_lengths, sorted_indice = torch.sort(lengths, descending=True)
        sorted_itemids = itemids[sorted_indice]
        sorted_valuenums = valuenums[sorted_indice]
        sorted_labels = labels[sorted_indice]
        sorted_timeids = timeids[sorted_indice]
        
        batch = {}
        batch['item_ids'] = sorted_itemids
        batch['value_nums'] = sorted_valuenums
        batch['time_ids'] = sorted_timeids
        batch['lengths'] = sorted_lengths
        batch['labels'] = sorted_labels
        
        return batch
    
    def _organize_batch(self, batch):
        instances = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return instances, labels   


def evaluation(dataloader, model, device):

    y_labels, y_scores = [], []
    for batch in dataloader:
            
        instances = {k:v.to(device) for k,v in batch.items() if k not in ['labels']}
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model(instances)

        y_labels += labels.cpu().tolist()
        y_scores += F.softmax(outputs, dim=1).cpu().tolist()

    y_labels = np.array(y_labels)
    y_scores = np.array(y_scores)

    try:
        auroc = roc_auc_score(y_labels, y_scores[:, 1])
    except ValueError:
        pass

    auprc = average_precision_score(y_labels, y_scores[:, 1])

    return auroc, auprc

f = open('./length_record.txt', 'r')
line = f.readline()
len_split = line.split(' ')
item_len = int(len_split[0])
value_len = int(len_split[1])
f.close()

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train = torch.LongTensor(np.load('./X_train_rnn.npy'))
    X_test = torch.LongTensor(np.load('./X_test_rnn.npy'))
    y_train = torch.LongTensor(np.load('./y_train.npy'))  
    y_test = torch.LongTensor(np.load('./y_test.npy'))

    model = GRU(item_len, value_len, 128, 128, 20, 1, 100).to(device)
    model.load_state_dict(torch.load('./gru_parameter.pt'))
    model.eval()

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, collate_fn=DataCollatorWithSorting())
    test_dataloader = DataLoader(test_dataset, collate_fn=DataCollatorWithSorting())

    train_auroc, train_auprc = evaluation(train_dataloader, model, device)
    test_auroc, test_auprc = evaluation(test_dataloader, model, device)

    text_to_record = str(20213207) + '\n' + str(train_auroc) + '\n' \
        + str(train_auprc) + '\n' + str(test_auroc) + '\n' + str(test_auprc)

    with open('./20213207_rnn.txt', 'w') as f:
        f.write(text_to_record)
    f.close()

if __name__ == '__main__':
    main()