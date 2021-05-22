import pickle
import time
import torch
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data.dataloader import DataLoader
from data.dataset import DischargeSummaryDataset
from data.data_collator import HFDataCollator
from utils.model import EndtoEndModel
from utils.eval import Evaluation


def main():

    wandb.init(project='HC_Assignment2', entity='swryu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='directory to save models', default='./trained_model_HC2.pt')
    parser.add_argument('--eval_result_dir', type=str, help='directory to record metrics', default='./student_id_model.txt')
    parser.add_argument('--cuda_num', type=int, help='cuda number', default=7)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--pre_config', type=str, help='pretrained config', default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--n_epochs', type=int, help='number of epochs', default=3)
    parser.add_argument('--assignment_mode', type=str, help='assignment mode', default='submit')
    # parser.add_argument('--lmbda', type=float, help='lambda for regularization', default=0.5)
    args = parser.parse_args()

    wandb.config.update(args)
    wandb.run.name = f"model:{args.pre_config} + Conv1D + Attention"

    device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(args.pre_config, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pre_config, config=config)
    model = EndtoEndModel(config, args.pre_config).to(device)
    data_collator = HFDataCollator(tokenizer)

    if args.assignment_mode != 'submit':
        print('-------------')
        print('Training mode')
        print('-------------')
        train_dataset = DischargeSummaryDataset(tokenizer, 'train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                collate_fn=data_collator,
                                                shuffle=True,
                                                drop_last=True)

        train_loss_list = []
        bce = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.n_epochs):
            train_loss = 0

            model.train()
            for batch_idx, (feature, targets) in enumerate(train_loader):
                feature = feature.to(device)
                targets = targets.to(device)

                probs = model(feature)
                loss = bce(probs, targets) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if batch_idx % 50 == 0:
                    print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f' 
                        %(epoch+1, args.n_epochs, batch_idx, len(train_loader), loss))

            loss_per_epoch = train_loss/len(train_loader)
            train_loss_list.append(loss_per_epoch)
            wandb.log({'Train Loss': loss_per_epoch})

        torch.save(model.state_dict(), args.model_dir)

    else:
        print('--------------')
        print('Inference mode')
        print('--------------')
        model.load_state_dict(torch.load(args.model_dir))
        test_dataset = DischargeSummaryDataset(tokenizer, 'test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.batch_size,
                                                  collate_fn=data_collator
                                                  )

        model.eval()
        auroc_macro, auroc_micro, auprc_macro, auprc_micro = Evaluation(test_loader, model, device)

        text_to_record = 'student_id' + '\n' + \
            str(auroc_macro) + '\n' + str(auroc_micro) + '\n' + \
            str(auprc_macro) + '\n' + str(auprc_micro)

        with open(args.eval_result_dir, 'w') as f:
            f.write(text_to_record)
        f.close()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time spent: {(end-start)/60} min.')