from os import linesep
from typing import Dict, Set
### You may import any Python's standard library here (Do not import other external libraries) ###


class Graph:
    def __init__(self, edges_path: str, states_path: str) -> None:
        ################## TODO: Fill out the class variables ##################
        #  self._nodes : Set of nodes                                          #
        #  self._neighbors : key: node, value: set of neighbors of node        #
        #  self._node_labels: key: node, value: label of node                  #
        # WARNING: Do not declare another class variables                      #
        ########################################################################
        
        self._nodes: Set[int] = set()
        self._neighbors: Dict[int, Set[int]] = dict()
        self._node_labels: Dict[int, int] = dict()
        
        with open(edges_path, 'r') as f, open(states_path, 'r') as g:
            for line in f.readlines():
                src, dst = [int(x) for x in line.strip().split()]
                self._nodes.update([src, dst])
                
                try:    
                    self._neighbors[src].add(dst)
                except:
                    self._neighbors[src] = set([dst])
                # Because it's "undirected" as stated in the condition of MRF. 
                try:
                    self._neighbors[dst].add(src)
                except:
                    self._neighbors[dst] = set([src])
                    
            for line in g.readlines():
                node_and_label = line.strip().split()
                if len(node_and_label) > 1:
                    node, label = [int(x) for x in node_and_label]
                    self._node_labels[node] = label
                
        f.close()
        g.close()
        ######################### Implementation end ###########################

    @property
    def nodes(self) -> Set[int]:
        return self._nodes

    @property
    def neighbors(self) -> Dict[int, Set[int]]:
        return self._neighbors

    @property
    def node_states(self) -> Dict[int, int]:
        return self._node_labels