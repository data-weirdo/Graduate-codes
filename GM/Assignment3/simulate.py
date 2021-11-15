import argparse
import numpy as np
import random
from SIR import sampleInitialNodes, runSimulation

import pickle
import time # I exceptionally added this module to compute actual computation time. :)

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--input", metavar="INPUT", type=str, default='graph.txt', help="a path containing input graph")
    parser.add_argument("--beta", metavar="BETA", type=float, default=0.1, help="the infection rate of the SIR model")
    parser.add_argument("--delta", metavar="DELTA", type=float, default=0.8, help="the recovery rate of the SIR model")
    parser.add_argument("--initial-nodes", metavar="INITIAL_NODES", type=int, default=50, help="a target number of initial nodes")
    parser.add_argument("--seed", metavar="SEED", type=int, default=0, help="a random seed")

    parser.add_argument('--problem_3_1', type=bool, default=False)
    parser.add_argument('--problem_3_2', type=bool, default=False)
    args=parser.parse_args()

    try:
        node_to_idx={}
        edges=set([])
        with open(args.input, "r") as f:
            for line in f:
                u, v = line.strip().split()
                if u == v: continue
                if u not in node_to_idx:
                    node_to_idx[u] = len(node_to_idx)
                if v not in node_to_idx:
                    node_to_idx[v] = len(node_to_idx)
                edges.add((node_to_idx[u], node_to_idx[v]))
                edges.add((node_to_idx[v], node_to_idx[u]))
            num_nodes = len(node_to_idx)
            edges = list(edges)
    except:
        print('ERROR: FILE DOES NOT EXIST!')
        exit(0)

    random.seed(args.seed)
    np.random.seed(args.seed)

    total_sum = 0

    if args.problem_3_1 == False and args.problem_3_2 == False:
        for i in range(100):
            initial_nodes = sampleInitialNodes(num_nodes, args.initial_nodes)
            total_sum += runSimulation(num_nodes, edges, initial_nodes, args.beta, args.delta)

    # To solve problem 3.1
    # python simulate.py --problem_3_1 True
    elif args.problem_3_1 == True:
        beta_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        recovered_list = []
        if args.problem_3_2 == True:
            raise ValueError('The anaylysis of problem3_1 and problem3_2 cannot be executed at the same time')
        else:
            initial_nodes = sampleInitialNodes(num_nodes, args.initial_nodes)
            for beta in beta_list:
                recovered = runSimulation(num_nodes, edges, initial_nodes, beta, args.delta)   
                recovered_list.append(recovered)
            with open('./problem3_1.pickle', 'wb') as f:
                pickle.dump([beta_list, recovered_list], f)
            f.close()

    # To solve problem 3.2
    # python simulate.py --problem_3_2 True
    else:
        initial_list = [10, 20, 50, 100, 200, 500]
        beta_delta_pair = [(0.005, 0.8), (0.02, 0.6)]
        if args.problem_3_1 == True:
            raise ValueError('The anaylysis of problem3_1 and problem3_2 cannot be executed at the same time')
        else:
            beta_delta_case1 = []
            beta_delta_case2 = []
            common_initial = []
            for initial_instance in initial_list:
                initial_nodes = sampleInitialNodes(num_nodes, initial_instance)
                for i, (beta, delta) in enumerate(beta_delta_pair):
                    recovered = runSimulation(num_nodes, edges, initial_nodes, beta, delta)
                    if i == 0:
                        common_initial.append(initial_instance)
                        beta_delta_case1.append(recovered)
                    else:
                        beta_delta_case2.append(recovered)
            with open('./problem3_2_1.pickle', 'wb') as f, \
                open('./problem3_2_2.pickle', 'wb') as g:
                pickle.dump([common_initial, beta_delta_case1], f)
                pickle.dump([common_initial, beta_delta_case2], g)
            f.close()

    # print('average influenced nodes:', total_sum / 100.0)