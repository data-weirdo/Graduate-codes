import dgl
import numpy as np
import torch
from copy import deepcopy
from typing import OrderedDict

def split_author_ids_and_make_list(row):
    result = row.tolist()[0].split(',')
    result = [int(x) for x in result]
    result = [str(x) for x in sorted(result)]
    return result

def get_mapped_input(pa_data, author_info):
    mapped_dict = OrderedDict()
    author_id_list = pa_data.to_dict()['author_id_list']

    for key, values in author_id_list.items():
        temp_mapped = []
        for value in values:
            mapped = author_info[value]
            temp_mapped.append(mapped)
        mapped_dict[key] = temp_mapped

    return mapped_dict

def make_api_input(edge):
    src_node = []
    dst_node = []

    for key, values in edge.items():
        dst_to_add = values.copy()
        src_to_add = [key]*len(dst_to_add)
        if len(dst_to_add) == 0:
            raise ValueError('Abnormal data')
        src_node.extend(src_to_add)
        dst_node.extend(dst_to_add)

    assert len(src_node) == len(dst_node), 'Length does not match. It is strange.'

    return src_node, dst_node

def edge_manipulation(g, sl, rv):
    if sl == True:
        g = dgl.add_self_loop(g)
    if rv == True:
        g = dgl.add_reverse_edges(g)
    return g

def map_to_bipartite_index(row, change_of_idx):
    value = row.values.tolist()
    value = [x.strip() if type(x) == str else x for x in value]
    flag = 0

    if len(value) == 3:
        tf_to_add = value[-1]
        flag = 1

    value = value[:2]
    new_value = [change_of_idx[str(x)] for x in value]    

    if flag == 1:
        new_value.append(tf_to_add)

    return new_value

def get_original_idx_back(edge_pair, mapping):
    edge_pair_cp = deepcopy(edge_pair)
    row, col = edge_pair.size()
    for i in range(row):
        for j in range(col):
            edge_pair_cp[i,j] = int(mapping[edge_pair[i,j].item()])    
    edge_pair_cp = edge_pair_cp.numpy().astype(str)
    return edge_pair_cp

def get_model_input_index(edge_pair, mapping):
    edge_pair_cp = np.zeros_like(edge_pair.astype(np.int))
    row, col = edge_pair.shape
    for i in range(row):
        for j in range(col):
            edge_pair_cp[i,j] = mapping[edge_pair[i,j]]
    edge_pair_cp = torch.tensor(edge_pair_cp)
    return edge_pair_cp