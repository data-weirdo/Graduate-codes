import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel

class EndtoEndModel(nn.Module):
    def __init__(self, config, model_args):
        super(EndtoEndModel, self).__init__()
        self.config = config
        self.model_args = model_args
        self.model_front = Bio_Clinical_Bert(self.config, self.model_args)
        self.model_back = nn.Sequential(
                Cnn1d(in_channels=self.config.max_position_embeddings, out_channels=self.config.hidden_size),
                Linear(self.config.hidden_size, 18449) # 18449: distinct number of ICD9 diagnosis + procedure codes 
            )

    def forward(self, x):
        output = self.model_front(x)
        output = self.model_back(output)
        output = torch.sigmoid(output)

        return output


class Bio_Clinical_Bert(PreTrainedModel):
    def __init__(self, config, model_args):
        super(Bio_Clinical_Bert, self).__init__(config)
        self.model_args = model_args
        self.config = AutoConfig.from_pretrained(self.model_args, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args, config=self.config)
        self.model = AutoModel.from_pretrained(self.model_args, config=self.config)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.model(**x)
        hidden = output[2]
        last_hidden = hidden[-1]

        return last_hidden


class Cnn1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Cnn1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn1d = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.model = nn.Sequential(
            self.conv1d,
            self.bn1d, 
            self.relu
        )

    def forward(self, x):
        x = self.model(x)
        output = attention(x, x, x)
        output = output.transpose(1, 2).contiguous().mean(dim=1)

        return output


class Linear(nn.Module):
    def __init__(self, in_features=None, out_features=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        output = self.linear(x)
        
        return output


def attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)
    p_attn = F.softmax(scores, dim=-1)
    
    return torch.matmul(p_attn, value)