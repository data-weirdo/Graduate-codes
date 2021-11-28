import os
import operator
from typing import Dict
from graph import Graph
from loopy_belief_propagation import loopy_belief_propagation


if __name__ == "__main__":
    ############################################################################
    # Each line of the edges file(polblogs_edges.txt, pubmed_edges.txt)        #
    # looks like:                                                              #
    #         <SOURCE NODE ID> <DESTINATION NODE ID>                           #
    # s.t. two integers are separated by space                                 #
    # ------------------------------------------------------------------------ #
    # Each line of the labels file(polblogs_labels.txt, pubmed_labels.txt)     #
    # looks like:                                                              #
    #         <NODE ID> <LABEL ID>   or   <NODE ID>                            #
    # s.t. <NODE ID> <LABEL ID> are separated by space                         #
    ############################################################################

    edges_path = os.path.join(os.getcwd(), "polblogs_edges.txt")
    labels_path = os.path.join(os.getcwd(), "polblogs_labels.txt")
    # edges_path = os.path.join(os.getcwd(), "pubmed_edges.txt")
    # labels_path = os.path.join(os.getcwd(), "pubmed_labels.txt")
    maxiters = 200
    tolerance = 1e-4

    # Initialize the graph
    graph: Graph
    graph = Graph(edges_path, labels_path)

    # Run the loopy belief propagation algorithm
    belief: Dict[int, Dict[int, float]]
    belief = loopy_belief_propagation(graph, maxiters, tolerance, edges_path)

    # Save the result file
    ################ TODO: Save the result of the algorithm ################
    #  result file name should  XXX_belief.txt                             #
    #  (XXX: name of graph, e.g., polblogs, pubmed)                        #
    #                                                                      #
    #  Each line of the result file looks like:                            #
    #   <NODE ID> <STATE ID>:<STATE BELIEF> <STATE ID>:<STATE BELIEF> ...  #
    # s.t. STATE ID and STATE BELIEF are separated by ":", and NODE ID and #
    # each <STATE ID>:<STATE BELIEF> pairs are separated by space          #
    ########################################################################
    
    lbp_result = []
    
    for node, belief in belief.items():
        record = str(node) + ' '
        for idx, state in enumerate(belief):
            record += f'{idx}:{round(state, 4)} '
        lbp_result.append(record.rstrip() + '\n')
        
    # To sort by node index
    src_node_idx = [int(x.split(' ')[0]) for x in lbp_result]
    src_sort_idx_temp = enumerate(src_node_idx)
    sort_index = [info[0] for info in sorted(src_sort_idx_temp, key=operator.itemgetter(1))]
    lbp_result = [lbp_result[x] for x in sort_index]
    
    file_name = edges_path.split('/')[-1].split('_')[0] + '_belief.txt'
        
    with open(file_name, 'w') as f:
        f.writelines(lbp_result)
    f.close()
    ######################### Implementation end ###########################

