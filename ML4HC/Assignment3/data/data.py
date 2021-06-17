import os
import pickle
import pandas as pd
import torch 
import torch.nn as nn
from collections import OrderedDict
from typing import List, OrderedDict
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

custom_transform = transforms.Compose([transforms.ToTensor()])              

class CXRDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, feature_path, label_path, transform=None, mode='test'):
    
        assert mode in ['train', 'valid', 'test'], 'Wrong mode argument'

        with open(feature_path, 'rb') as f:
            img = pickle.load(f)
        f.close()
 
        with open(label_path, 'r') as f:
            label = f.readlines()
        f.close()  
        
        self.transform = transform
        img = self.image_file_processing(img)
        label = self.label_file_processing(label)

        self.img = img # List
        self.label = label # List

        threshold = round(0.95 * len(self.label))
        if mode == 'train':
            self.img = self.img[:threshold]
            self.label = self.label[:threshold]
        elif mode == 'valid':
            self.img = self.img[threshold:]
            self.label = self.label[threshold:]

    def image_file_processing(self, feature: OrderedDict) -> List:
        image_list = []
        for _, value in feature.items():
            value = self.transform(value)
            image_list.append(value)
        return image_list

    def label_file_processing(self, label: List) -> List[List]:
        label_list = []
        for index, contents in enumerate(label):   
            contents = contents.split(',')[1:]
            contents[-1] = contents[-1][0]
            contents = list(map(lambda x: int(x), contents))
            label_list.append(torch.FloatTensor(contents))
        return label_list 

    def __getitem__(self, index):
        img = self.img[index]
        label = self.label[index]
        return (img, label)

    def __len__(self):
        return len(self.img)                             


class CXRDataLoader:
    def __init__(self, train_dset, valid_dset, test_dset, batch_size, n_workers, mode):
        assert mode in ['train', 'test'], 'You got an wrong mode on parser.'

        if mode == 'train':
            self.train_dset = train_dset
            self.valid_dset = valid_dset
        else: 
            self.test_dset = test_dset
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.mode = mode

    def customloader(self):
        if self.mode == 'train':
            train_loader = DataLoader(dataset=self.train_dset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.n_workers,
                                    drop_last=True)   

            valid_loader = DataLoader(dataset=self.valid_dset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=self.n_workers, 
                                    drop_last=True)   

            return train_loader, valid_loader

        else:
            test_loader = DataLoader(dataset=self.test_dset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=self.n_workers, 
                                    drop_last=True)   

            return test_loader
