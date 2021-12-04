import os
import logging
import pprint
import tqdm
import torch
import torch.nn as nn
import numpy as np
import models 
import pickle
import random
from time import time
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from datasets import (
    PaperAuthorDataset,
    AuthorsIDDataset,
    PaperAuthorIDPairsDataset
)
from utils.dataset_utils import get_original_idx_back, get_model_input_index
from utils.model_utils import *
from utils.trainer_utils import (
    rename_logger,
    should_stop_early
)
from sortedcontainers import SortedSet

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.data_loaders = dict()

        # print args
        logger.info(pprint.pformat(vars(args)))

        model = models.build_model(args)
        print('model loaded!')

        logger.info(model)
        logger.info("model: {}".format(model.__class__.__name__))
        logger.info(
            "num. model params: {:,} (num. trained: {:,})".format(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
        )

        # self.model = nn.DataParallel(model, device_ids=args.device_ids).to('cuda')
        self.model = model.to(self.args.device)
        self.pair_in_tvq = [] # tvq: train, valid, query

        for subset in ['paper_author', 'train'] + self.args.valid_subsets:
            self.load_dataset(subset)
        self.load_dataset('author_pair')     

        self.pair_in_tvq = get_original_idx_back(torch.vstack(self.pair_in_tvq), self.idx2author).tolist()

        # if os.path.isfile(os.path.join(self.args.current_dir, 'paper_author_id_pairs_extracted.pkl')):
        #     with open(os.path.join(self.args.current_dir, 'paper_author_id_pairs_extracted.pkl'), 'rb') as f:
        #         self.all_pairs = pickle.load(f)
        #     f.close()
        # else:
        #     self.extract_non_overlap_id_pairs()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = lambda epoch: epoch / 200 if epoch < 200 \
                    else (1 + np.cos((epoch-200) * np.pi / (self.args.epochs - 200))) * 0.5
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = scheduler)
        self.criterion = nn.CrossEntropyLoss()

    def load_dataset(self, split: str):
        assert split in ['paper_author', 'train', 'valid', 'query', 'author_pair'], 'Not allowed data!'

        self.batch_size = {
                'train': self.args.train_batch_size,
                'valid': self.args.valid_batch_size,
                'query': self.args.valid_batch_size,
                'author_pair': self.args.valid_batch_size
            }

        if split == 'paper_author':
            self.paper_author_graph, self.author2idx, self.idx2author, self.all_pairs = PaperAuthorDataset(
                input_path=self.args.input_path,
                pair_path=self.args.current_dir,
                self_loop=self.args.self_loop,
                reverse=self.args.reverse,
                embedding_dim=self.args.embedding_dim
            ).processed()

            self.paper_author_graph = self.paper_author_graph.to(self.args.device)

        elif split == 'author_pair':
            author_pairs = PaperAuthorIDPairsDataset(
                self.all_pairs
            )

            self.data_loaders[split] = DataLoader(
                author_pairs, 
                collate_fn=author_pairs.collator, 
                batch_size=self.batch_size[split], 
                num_workers=4, 
                shuffle=False
            )

        else:
            author_ids_dataset = AuthorsIDDataset(
                input_path=self.args.input_path, 
                author2idx=self.author2idx, 
                split=split, 
                reference_index=self.idx2author,
                negative_sample=self.args.negative_sample,
                pair_gather=self.pair_in_tvq
            )

            shuffle = True if split == 'train' else False

            self.data_loaders[split] = DataLoader(
                author_ids_dataset, 
                collate_fn=author_ids_dataset.collator, 
                batch_size=self.batch_size[split], 
                num_workers=4, 
                shuffle=shuffle
            )

    def extract_non_overlap_id_pairs(self):
        self.overlapping_mapped = get_original_idx_back(self.pair_in_train_tvq, self.idx2author)
        self.overlapping_mapped = list(tuple(map(tuple, self.overlapping_mapped)))
        self.all_pairs = [x for x in self.all_pairs if x not in self.overlapping_mapped] # It takes infinite amount of time..
        with open(os.path.join(self.args.current_dir, 'paper_author_id_pairs_extracted.pkl'), 'wb') as f:
            pickle.dump(self.overlapping_mapped, f)
        f.close()

    def train(self, wandb, run):  

        for epoch in range(1, self.args.epochs + 1):

            logger.info(f"begin training epoch {epoch}")
 
            train_loss = 0.0
            train_acc = 0.0
            augmentation = None
            # augmentation = Augmentation(*self.args.aug_params)
            # augmented1, augmented2 = augmentation(self.paper_author_graph, self.args.device)
            
            self.model.train()
            print('trainer loop start')
            for _ , sample in enumerate(tqdm.tqdm(self.data_loaders['train'])):
                augmentation = Augmentation(*self.args.aug_params)
                augmented1, augmented2 = augmentation(self.paper_author_graph, self.args.device)
                
                loss1, super_logit = self.model(augmented1, 
                                                augmented2, 
                                                sample['edge_pair'], 
                                                self.args.device, 
                                                'train')
                target = sample['label'].view(-1).to(self.args.device)

                self.optimizer.zero_grad(set_to_none=True)

                loss2 = self.criterion(super_logit, target.to(self.args.device))
                loss = loss1 + self.args.beta * loss2
                loss.backward(retain_graph=True)

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                self.scheduler.step()
                self.model.update_moving_average()

                train_loss += loss.item() 

                wandb.log({'Train Loss': loss})

                with torch.no_grad():
                    pred = torch.argmax(super_logit[0:super_logit.size(0)//(self.args.negative_sample+1)], -1)
                    target = target[0:super_logit.size(0)//(self.args.negative_sample+1)]
                    correct = (target == pred).sum().detach().cpu()
                    acc = int(correct) / len(pred)
                    train_acc += acc
                    
            # This part actually degrades the performance of the model.
            # self.paper_author_graph.ndata['x'] = augmented1.ndata['x']

            avg_train_loss = train_loss / len(self.data_loaders['train'])
            avg_train_acc = train_acc / len(self.data_loaders['train'])

            wandb.log({'Train Epoch': epoch,
                    'Avg Train Loss': avg_train_loss,
                    'Avg Train Acc.': avg_train_acc})

            with rename_logger(logger, "train"):
                logger.info(
                    "epoch: {}, loss: {:.3f}, acc: {:.3f}".format(
                        epoch, avg_train_loss, avg_train_acc)
                )

            if epoch > 10:
                should_stop = self.validate_and_save(epoch, ['valid'], wandb, run)
                if should_stop:
                    _ = self.validate(0, ['query'], wandb, run)
                    _ = self.validate(0, ['author_pair'], None, run)
                    break

    def validate(
        self,
        epoch,
        valid_subsets, 
        wandb,
        run, 
    ):

        for subset in valid_subsets:

            logger.info("begin validation on '{}' subset".format(subset))

            # Only used at 'valid'
            valid_loss = 0.0
            valid_acc = 0.0
            
            # Only used at 'query'
            query_pred = []
            label_list = []

            # Only used at 'author_pair'
            pred_true_list = []

            self.model.eval()
            for _, sample in enumerate(self.data_loaders[subset]):
                with torch.no_grad():
                    if subset in ['valid', 'query']:
                        super_logit = self.model(self.paper_author_graph, 
                                                None, 
                                                sample['edge_pair'], 
                                                self.args.device, 
                                                subset)
                        
                        pred = torch.argmax(super_logit, -1)

                        if subset == 'valid':
                            target = sample['label'].view(-1).to(self.args.device)

                            self.optimizer.zero_grad(set_to_none=True)

                            loss = self.criterion(super_logit, target.to(self.args.device))
                            valid_loss += loss.item() 

                            wandb.log({'Valid Loss': loss})

                            correct = (pred == target).sum().detach().cpu()
                            acc = int(correct) / len(target)

                            valid_loss += loss.item()
                            valid_acc += acc

                        else:
                            original_edge_pair = get_original_idx_back(sample['edge_pair'], self.idx2author)
                            pred = pred.cpu().numpy().astype(str)
                            concat = np.hstack((original_edge_pair, pred.reshape(-1,1))).tolist()
                            query_pred.extend(concat)
                            label_list.extend(pred.tolist())

                    else:
                        mapped_sample = get_model_input_index(sample, self.author2idx)
                        super_logit = self.model(self.paper_author_graph, 
                                                None, 
                                                mapped_sample, 
                                                self.args.device, 
                                                subset)

                        pred = torch.argmax(super_logit, -1)
                        index = (pred == 1).nonzero().view(-1).cpu().numpy()
                        selected_sample = sample[index, :].tolist()
                        pred_true_list.extend(selected_sample)

            if subset == 'valid':
                avg_valid_loss = valid_loss / len(self.data_loaders[subset])
                avg_valid_acc = valid_acc / len(self.data_loaders[subset])

                wandb.log({'Avg Valid Loss': avg_valid_loss, 
                        'Avg Val Acc.': avg_valid_acc})

                with rename_logger(logger, subset):
                    logger.info(
                        "epoch: {}, acc: {:.3f}".format(
                            epoch, avg_valid_acc
                        )
                    )

            elif subset == 'query':
                query_pred = [', '.join(x) for x in query_pred]
                label_list = [int(x) for x in label_list]

                os.makedirs(self.args.answer_path, exist_ok=True)
                with open(os.path.join(self.args.answer_path, 'query_answer.csv'), 'w') as f:
                    for line in query_pred:
                        f.write(line)
                        f.write('\n')
                f.close()
                wandb.log({'Query dataset same ID pair cnt': sum(label_list)})

                avg_valid_acc = None

            else:
                pred_true_list = [', '.join(x) for x in pred_true_list]
                with open(os.path.join(self.args.answer_path, 'same_author.csv'), 'w') as f:
                    for line in pred_true_list:
                        f.write(line)
                        f.write('\n')
                f.close()
                
                avg_valid_acc = None

        return avg_valid_acc

    def validate_and_save(
        self,
        epoch,
        valid_subsets, 
        wandb, 
        run
    ):
        os.makedirs(self.args.save_dir, exist_ok=True)
        if (self.args.disable_save 
         ):
            return False
        if (
            self.args.disable_validation
            or valid_subsets is None
        ):
            logger.info(
                "Saving checkpoint to {}".format(
                    os.path.join(self.args.
                    save_dir, f"epoch{epoch}_last.pt")
                )
            )
            torch.save(
                {
                    'model_state_dict': self.model.module.state_dict() if (
                        isinstance(self.model, DataParallel)
                    ) else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epoch,
                    'args': self.args
                },
                os.path.join(self.save_dir, f"epoch{epoch}_last.pt")
            )
            logger.info(
                "Finished saving checkpoint to {}".format(
                    os.path.join(self.save_dir, f"epoch{epoch}_last.pt")
                )
            )
            return False

        should_stop = False

        valid_acc = self.validate(epoch, valid_subsets, wandb, run)
        should_stop |= should_stop_early(self.args.patience, valid_acc)

        prev_best = getattr(should_stop_early, "best", None)
        if (
            self.args.patience <= 0
            or prev_best is None
            or (prev_best and prev_best == valid_acc)
        ):
            logger.info(
                "Saving checkpoint to {}".format(
                    os.path.join(self.args.save_dir, f"epoch{epoch}_best.pt")
                )
            )
            torch.save(
                {
                    'model_state_dict': self.model.module.state_dict() if (
                        isinstance(self.model, DataParallel)
                    ) else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epoch,
                    'args': self.args,
                },
                os.path.join(self.args.save_dir, f"epoch{epoch}_best.pt")
            )
            logger.info(
                "Finished saving checkpoint to {}".format(
                    os.path.join(self.args.save_dir, f"epoch{epoch}_best.pt")
                )
            )

        return should_stop