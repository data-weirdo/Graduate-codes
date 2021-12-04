import os
import logging
import pickle
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import pandas as pd
import dgl

from copy import deepcopy
from collections import OrderedDict
from itertools import chain, combinations
from utils.dataset_utils import *

logger = logging.getLogger(__name__)


class PaperAuthorDataset(object):
    def __init__(self, 
                input_path, 
                pair_path,
                self_loop, 
                reverse, 
                embedding_dim
                ):  

        input_path = os.path.join(input_path, 'paper_author_relationship.csv')

        data = self.get_pd_dataframe(input_path)
        paper_ids = data.index.values.tolist()

        """
        Explanation for Annotated (#) Part:
        It takes so much time for considering all the combinations of pair ids at solving the question of assignment. 
        So, although I made this code for pre-processing, as this part takes so much time, 
        I killed this part in actual implementation step, and saved the data in the other place. 

        # of pairs from combination: 1887528961
        """
        # For ID-Pair prediction task using the IDs which appear in paper_author_relationship.csv
        # I save the pairs on the fly, because it takes so much time to make combination dataset!
        if os.path.isfile(os.path.join(pair_path, 'paper_author_id_pairs.pkl')):
            with open(os.path.join(pair_path, 'paper_author_id_pairs.pkl'), 'rb') as f:
                self.all_pairs = pickle.load(f)
            f.close()

        else:
            all_pairs = list(chain(*list(chain(*data.values.tolist()))))
            all_pairs = list(set(all_pairs))
            self.all_pairs = list(combinations(all_pairs, 2))
            with open(os.path.join(pair_path, 'paper_author_id_pairs.pkl'), 'wb') as f:
                pickle.dump(self.all_pairs, f)
            f.close()

        self.author2idx, self.idx2author = self.edge_idx_align(data, paper_ids)
        aligned_edge = get_mapped_input(data, self.author2idx)
        src_nodes, dst_nodes = make_api_input(aligned_edge)

        total_row_length = len(paper_ids) + len(self.author2idx)
        paper_embedding = nn.Embedding(total_row_length, embedding_dim)

        self.graph = dgl.graph((torch.LongTensor(src_nodes), torch.LongTensor(dst_nodes)))
        self.graph = edge_manipulation(self.graph, self_loop, reverse)
        self.graph.ndata['x'] = paper_embedding(torch.tensor(list(range(total_row_length))))

    def get_pd_dataframe(self, file_path):
        data = pd.read_csv(file_path, sep='\t', header=None)
        data.columns = ['author_id']
        data['author_id_list'] = data.apply(split_author_ids_and_make_list, axis=1)
        del data['author_id']
        return data
        
    def edge_idx_align(self, data, paper_ids):
        author_ids_to_integer = OrderedDict()
        integer_to_author_ids = OrderedDict()
        cnt = 1

        author_ids = [x.tolist()[0] for x in data.values]
        max_paper_ids = max(paper_ids)

        for authors_per_paper in author_ids:
            for author_id in authors_per_paper:
                if author_id not in author_ids_to_integer.keys():
                    author_ids_to_integer[author_id] = cnt + max_paper_ids
                    cnt += 1
                
        for key, value in author_ids_to_integer.items():
            integer_to_author_ids[value] = key
        return author_ids_to_integer, integer_to_author_ids

    def processed(self):
        return self.graph, self.author2idx, self.idx2author, self.all_pairs
        

class AuthorsIDDataset(torch.utils.data.Dataset):
    def __init__(self, 
                input_path, 
                author2idx, 
                split, 
                reference_index, 
                negative_sample, 
                pair_gather):

        self.split = split
        self.label = None

        self.input_path = input_path
        self.data = self.open_data(split)
        self.data['processed'] = self.data.apply(map_to_bipartite_index, change_of_idx=author2idx, axis=1)
        edge_label = self.data['processed'].values.tolist()
        self.reference_idx = reference_index
        self.negative_sample = negative_sample

        self.edge_pair = torch.tensor([x[:2] for x in edge_label]).view(-1, 2)
        pair_gather.append(self.edge_pair)
        
        if self.split != 'query':
            self.label = torch.tensor([int(bool(x[-1])) for x in edge_label]).view(-1, 1)

    def open_data(self, split):
        assert split in ['train', 'valid', 'query']
        data = pd.read_csv(os.path.join(self.input_path, f'{split}_dataset.csv')) # Not splitted
        if split in ['train', 'valid']:
            data.columns = ['ID1', 'ID2', 'label']
        else:
            data.columns = ['ID1', 'ID2']
        return data

    def __len__(self):
        return len(self.edge_pair)
    
    def __getitem__(self, index):
        edge_pair = self.edge_pair[index]
        if self.split != 'query':
            label = self.label[index]
            return {
                'edge_pair': edge_pair, 
                'label': label
            }
        else:
            return {
                'edge_pair': edge_pair
            }

    def collator(self, samples):
        final_input = dict()
        final_input['edge_pair'] = torch.vstack([sample['edge_pair'] for sample in samples])

        if self.split != 'query':
            final_input['label'] = torch.vstack([sample['label'] for sample in samples])

        if self.split == 'train':
            """
            To add 'negative samples' for effective training
            """
            author_indices = list(self.reference_idx.keys())
            
            base = final_input['edge_pair'][:,0].numpy().repeat(self.negative_sample).reshape(-1,1)
            sampled = np.random.choice(author_indices, len(base), replace=True).reshape(-1,1)
            negative_x = torch.tensor(np.hstack((base, sampled)))
            final_input['edge_pair'] = torch.vstack((final_input['edge_pair'], negative_x))

            negative_y = torch.tensor(np.zeros((len(base),)).reshape(-1,1))
            final_input['label'] = torch.vstack((final_input['label'], negative_y)).to(int)            

        return final_input


class PaperAuthorIDPairsDataset(torch.utils.data.Dataset):
    def __init__(self, pair_data):
        self.pair_data = pair_data

    def __len__(self):
        return len(self.pair_data)
    
    def __getitem__(self, index):
        pair_instance = self.pair_data[index]
        return {'pair': pair_instance}

    def collator(self, samples):
        pairs = np.array([list(sample['pair']) for sample in samples])
        return pairs