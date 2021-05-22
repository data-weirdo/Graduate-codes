import sys
import os
import argparse
import wandb
import torch
import torch.nn as nn
import wandb

from torch.utils.data import Dataset, DataLoader
from data.data_collator import CelebaDataset, CustomDataLoader, custom_transform
from src import *
from utils import *

def main():

    wandb.init(project='DL_Assignment1', entity='swryu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_num', type=int, help='cuda number', default=1)
    parser.add_argument('--model', type=int, help='model number', default=1, 
                        choices = [1, 2, 3]) # 1: MLP, 2: VGG19, 3: ResNet-18
    parser.add_argument('--reg', type=int, help='regularization number', default=0, 
                        choices = [0, 1, 2, 3]) # 0: No regularization, 1: Dropout, 2: L2 norm, 3: L1 norm 
    parser.add_argument('--optimizer', type=int, help='optimizer', default=1, 
                        choices = [1, 2, 3]) # 1: SGD with momentum, 2: AdaGrad, 3: Adam
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--drop_param', type=float, help='dropout forget rate', default=0)
    # parser.add_argument('--lmda', type=float, help='hyperparam for L1, L2 regularization', default=1e-4)
    # parser.add_argument('--step_size', type=int, help='step size for StepLR', default=3)
    
    args = parser.parse_args()

    wandb.config.update(args)
    args_model, args_reg, args_optimizer = return_name(args.model, args.reg, args.optimizer)
    wandb.run.name = f"{args_model}-{args_reg}-{args_optimizer}"

    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    # args.train_path => './data/celeba-gender-train.csv'
    train_dataset = CelebaDataset(csv_path='./data/celeba-gender-train.csv',
                                  img_dir='./img_align_celeba_unzipped/',
                                  transform=custom_transform)

    valid_dataset = CelebaDataset(csv_path='./data/celeba-gender-valid.csv',
                                img_dir='./img_align_celeba_unzipped/',
                                transform=custom_transform)

    test_dataset = CelebaDataset(csv_path='./data/celeba-gender-test.csv',
                                img_dir='./img_align_celeba_unzipped/',
                                transform=custom_transform)
                            
    train_loader, valid_loader, test_loader, train_length, valid_length, test_length  = CustomDataLoader(train_dataset,\
        valid_dataset, test_dataset, args.batch_size, args.n_workers).customloader()

    trained_model = trainer(args, train_loader, train_length, valid_loader, valid_length, device, wandb)
    inference(args, trained_model, test_loader, test_length, device, wandb)

    wandb.finish()

if __name__ == '__main__':
    main()