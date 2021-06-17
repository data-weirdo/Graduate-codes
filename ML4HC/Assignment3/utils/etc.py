import random
import torch
import pickle
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def get_weights(x): # x <= label_list
    x = np.array(x)
    train_nrow, train_ncol = x.shape
    train_whole = [train_nrow] * train_ncol
    train_positive = np.sum(x, axis=0)
    train_negative = np.subtract(np.array(train_whole), np.array(train_positive))
    train_weight_to_use = 1 / (train_positive / train_negative)
    weight = torch.FloatTensor(train_weight_to_use)

    return weight

def calculate_pos_weights(label_path):
    with open(label_path, 'r') as f:
        label = f.readlines()
    f.close()

    label_list = []
    for index, contents in enumerate(label):   
        contents = contents.split(',')[1:]
        contents[-1] = contents[-1][0]
        contents = list(map(lambda x: int(x), contents))
        label_list.append(contents)

    weights = get_weights(label_list)
        
    with open('./pos_weight.pkl', 'wb') as f:
        pickle.dump(weights, f)
    f.close()

    return weights