  
import sys
import os
import argparse
import wandb
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from data.data import CXRDataset, CXRDataLoader, custom_transform
from src import *
from utils import *
from torchvision import models

def main():
    wandb.init(project='HC_Assignment3', entity='swryu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_num', type=int, help='cuda number', default=0)
    parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--model_dir', type=str, default='./model_param.pt')
    parser.add_argument('--out_dir', type=str, default='./20213207_model.txt')
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        wandb.config.update(args)
        wandb.run.name = 'Resnet50'
        seed_everything(args.seed)

        train_dataset = CXRDataset(feature_path='./X_train.pkl',
                                    label_path='./y_train.txt',
                                    transform=custom_transform,
                                    mode='train')

        valid_dataset = CXRDataset(feature_path='./X_train.pkl',
                                    label_path='./y_train.txt',
                                    transform=custom_transform,
                                    mode='valid')

        pos_weights = calculate_pos_weights('y_train.txt') # To solve class imbalance

        train_loader, valid_loader = CXRDataLoader(train_dataset, 
                                                   valid_dataset,
                                                   None, 
                                                   args.batch_size, 
                                                   args.n_workers,
                                                   'train').customloader()

        trainer(args, train_loader, len(train_dataset), valid_loader, len(valid_dataset), device, wandb, pos_weights)
        wandb.finish()

    else:
        trained_model = ResNet50(args.dropout_rate)
        # trained_model = models.resnet50(pretrained=True)
        test_dataset = CXRDataset(feature_path='./X_test.pkl',
                                label_path='./y_test.txt',
                                transform=custom_transform,
                                mode='test')
                                
        test_loader = CXRDataLoader(None,
                                    None,
                                    test_dataset, 
                                    args.batch_size, 
                                    args.n_workers,
                                    'test').customloader()

        inference(args, trained_model, test_loader, device)


if __name__ == '__main__':
    main()
