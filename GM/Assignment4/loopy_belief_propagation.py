from typing import Dict, List
from graph import Graph
from collections import defaultdict
from copy import deepcopy
from itertools import chain

def loopy_belief_propagation(
    graph: Graph,
    max_iters: int,
    tol: float,
    edges_path: str
) -> Dict[int, Dict[int, List]]:
    belief: Dict[int, Dict[int, List]] = defaultdict(dict)
    ####### TODO: Implement the loopy belief propagation algorithm #########
    #  belief : key: node, value: dictionary in which the key is state and # 
    #  the value is belief of state                                        #
    ########################################################################
    state_cnt = 2 if 'polblogs' in edges_path else 3

    # message initialization
    messages = defaultdict(dict)
    for src, dst in graph._neighbors.items():
        for dst_instance in dst:
            messages[src].update({dst_instance: [1]*state_cnt})
            belief[src].update({dst_instance: [0]*state_cnt})
    
    # For Message update 
    for itr in range(max_iters):
        delta: float = 0.0
        new_messages = deepcopy(messages)    
    
        for src in list(graph._nodes):
            dsts = graph._neighbors[src]
            
            node_potential = [make_node_potential(edges_path, graph, src) for _ in range(state_cnt)] # fixed
            edge_potential = make_edge_potential(edges_path) # fixed
            
            numerator = [messages[dst][src] for dst in dsts]
            numerator = list(zip(*numerator))
            numerator = [multiplyList(x) for x in numerator]
            numerator = [numerator for _ in range(state_cnt)]
            
            denominators = [[messages[dst][src] for _ in range(state_cnt)] for dst in dsts]
            
            for idx, dst in enumerate(dsts):
                denominator = denominators[idx]
                
                for i in range(state_cnt):
                    np_instance = node_potential[i]
                    ep_instance = edge_potential[i]
                    numer_instance = numerator[i]
                    denom_instance = denominator[i]
                    
                    new_messages[src][dst][i] = calculate_messages(np_instance, ep_instance, numer_instance, denom_instance)
                
                new_messages[src][dst] = normalize(new_messages[src][dst])
        
        previous = [list(chain(*list(x.values()))) for x in list(messages.values())]
        previous = list(chain(*previous))
        
        current = [list(chain(*list(x.values()))) for x in list(new_messages.values())]
        current = list(chain(*current))

        delta = sum([abs(x - y) for x, y in zip(previous, current)])
        print(f"[Iter {itr}]\tDelta = {delta}")
        
        #### Check the convergence ###
        # Stop the iteration if L1norm[message(t) - message(t-1)] < tol
        if delta < tol:
            break
    
        messages = new_messages

    # For belief update
    for src in list(graph._nodes):
        dsts = graph._neighbors[src]
    
        node_potential = make_node_potential(edges_path, graph, src)
        
        numerator = [messages[dst][src] for dst in dsts]
        numerator = list(zip(*numerator))
        numerator = [multiplyList(x) for x in numerator]
        
        belief_instance = [multiplyList(x) for x in list(zip(*(node_potential, numerator)))]
        belief_instance = normalize(belief_instance)
        
        belief[src] = belief_instance
    ######################### Implementation end #########################
    return belief


def normalize(msg):
    denominator = sum(msg)
    normalized = [x / denominator for x in msg]
    return normalized 


def calculate_messages(np, ep, numer, denom):
    message = 0.0
    zipped = list(zip(*[np, ep, numer, denom]))
    for n, e, nu, de in zipped:
        value = n * e * nu / de
        message += value
    return message


def multiplyList(myList) :
    result = 1
    for x in myList:
         result = result * x
    return result


def make_node_potential(edges_path, graph, key_idx):
    if 'polblogs' in edges_path:
        try:
            label = graph._node_labels[key_idx]
            if label == 0:
                result = [0.9, 0.1]
            else:
                result = [0.1, 0.9]
        except:
            result = [0.5, 0.5]
            
    elif 'pubmed' in edges_path:
        try:
            label = graph._node_labels[key_idx]
            if label == 0:
                result = [0.9, 0.05, 0.05]
            elif label == 1:
                result = [0.05, 0.9, 0.05]
            else:
                result = [0.05, 0.05, 0.9]
        except:
            result = [0.33, 0.33, 0.33]
            
    else:
        raise ValueError('You entered the wrong data path')
    
    return result
    
    
def make_edge_potential(edges_path):
    if 'polblogs' in edges_path:
        adj = [[0.501, 0.499], 
               [0.499, 0.501]]
    elif 'pubmed' in edges_path:
        adj = [[0.334, 0.333, 0.333], 
               [0.333, 0.334, 0.333],
               [0.333, 0.333, 0.334]]
    else:
        raise ValueError('You entered the wrong data path')
    
    return adj