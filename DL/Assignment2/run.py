import pickle
import time
import torch
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from data.process import prepare_train_features, prepare_validation_features
from utils.etc import seed_everything, wandb_logging_name, model_selection
from utils.eval import calculate_score
from src.trainer import train, evaluate
from transformers import AutoTokenizer, default_data_collator

def main():
    start_time = time.time()
    wandb.init(project='DL_Assignment2', entity='swryu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_num', type=int, help='cuda number', default=4)
    parser.add_argument('--model', type=int, help='model number', default=3, choices = [1, 2, 3]) # 1: LSTM, 2: Transformer, 3: BERT
    parser.add_argument('--model_checkpoint', type=str, 
                        help='model check point which will be only used for BERT', 
                        default = 'google/bert_uncased_L-4_H-256_A-4')
    parser.add_argument('--hidden_dim', type=int, 
                        help='hidden dimension side of embedding matrix',
                        default = 768)
    parser.add_argument('--num_layers', type=int, default=3,
                        help='# of leayrs in transformer')
    parser.add_argument('--head_size', type=int, 
                        help='multi-head attention head size',
                        default = 12)
    parser.add_argument('--datasets', type=str, default='squad')
    parser.add_argument('--batch_size', type=int, help='batch size', default=12)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)

    args = parser.parse_args()

    # logging args
    wandb.config.update(args)
    model_name = wandb_logging_name(args.model)
    wandb.run.name = f"{model_name}-in-QA"

    # system args
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    # data args
    datasets = load_dataset(args.datasets)
    trained_datasets = datasets.map(prepare_train_features, batched=True, remove_columns = datasets['train'].column_names)
    valid_datasets = datasets['validation'].map(prepare_validation_features, batched=True, remove_columns = datasets['validation'].column_names)
    
    train_dataset = trained_datasets['train']


    train_loader = DataLoader(train_dataset, 
                            batch_size = args.batch_size,
                            collate_fn = default_data_collator, 
                            shuffle = True)
    
    valid_loader = DataLoader(valid_datasets.remove_columns('offset_mapping'),
                            batch_size = args.batch_size,
                            collate_fn = default_data_collator,
                            shuffle = False)
                        
    model = model_selection(args.model, 
                            model_checkpoint = args.model_checkpoint, 
                            dim = args.hidden_dim,
                            num_heads = args.head_size,
                            num_layers = args.num_layers)

    trained_model = train(args, model, train_loader, device, wandb)
    raw_predictions = evaluate(trained_model, valid_loader, device, wandb)
    calculate_score(datasets, valid_datasets, raw_predictions, wandb)

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    wandb.log({'Total time spent (minute)': total_time})
    
    wandb.finish()

if __name__ == '__main__':
    main()