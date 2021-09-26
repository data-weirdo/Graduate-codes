import argparse
import time
import numpy as np
from scipy import sparse
from typing import List

def genRMat(N, E, probs, output_path):
    print('numNodes:', N)
    print('numEdges:', E)
    print('probabilites: (pA, pB, pC, pD) =', '(' + ','.join(map(str, probs)) + ')')
    print('output_path:', output_path)
    matrix = np.zeros((N, N), dtype=np.int64)
    
    ### TODO: WRITE YOUR CODE HERE. YOU SHOULD *NOT* MODIFY THE OTHER PARTS! ###    
    def _get_next_idx(idx_info: List, 
                    partition_num: int):
        """
        Explanation for the variables' name below:
        r: row, c: column
        s: start, e: end, m: middle
        """
        rs, re, cs, ce = idx_info
        rm = (re + rs) // 2
        cm = (ce + cs) // 2
        if partition_num == 0: # a
            idx_info = [rs, rm, cs, cm]
        elif partition_num == 1: # b
            idx_info = [rs, rm, cm, ce]
        elif partition_num == 2: # c
            idx_info = [rm, re, cs, cm]
        else: # d
            idx_info = [rm, re, cm, ce]
        return idx_info

    def _recursive_mat(matrix: np.array, 
                    probs: List, 
                    idx_info: List, 
                    total_edge_cnt: int):
        # When satisfying 1x1 criteria:
        if idx_info[1]-idx_info[0] == 1: # 1x1 size block obtained
            row_index = idx_info[0]
            col_index = idx_info[2]
            if matrix[row_index, col_index] == 0:
                matrix[row_index, col_index] = 1
                total_edge_cnt += 1
            return matrix, total_edge_cnt

        # Iterations still left:
        else:
            partition = np.random.choice(len(probs), 1, p=probs)[0]
            idx_info = _get_next_idx(idx_info, partition)
            return _recursive_mat(matrix, probs, idx_info, total_edge_cnt)

    cum_edge_cnt = 0 # Edge count accumulation 
    idx_info = [0, N, 0, N]
    while True:
        matrix, cum_edge_cnt = _recursive_mat(matrix, probs, idx_info, cum_edge_cnt)
        if cum_edge_cnt == E:
            break

    matrix = sparse.csr_matrix(matrix)
    sparse.save_npz(output_path, matrix)
    ############################################################################
    
    # Validation for outputs. DO *NOT* MODIFY THIS PART!
    valid = True
    if matrix.dtype != np.int64:
        print(f'ERROR: Wrong datatype. (Expected: np.int64, but found {matrix.dtype}.)')
        valid = False
    if matrix.shape != (N, N):
        print(f'ERROR: Wrong shape. (Expected: ({N}, {N}), but found {matrix.shape}.)')
        valid = False
    if matrix.min() != 0 or matrix.max() != 1:
        print(f'ERROR: The output should be nonnegative, and should be binary.')
        valid = False
    if matrix.sum() != E:
        print(f'ERROR: The number of non-zero entries in the output shoule be {E}, but found {matrix.sum()}')
        valid = False
        
    if valid:
        print(f'Pretest passed! However, it does not guarantee the correctness of your code.')
        try:
            sparse.save_npz(output_path, matrix)
        except:
            print(f'Failed to save the output. Please check the output path again.')
    else:
        print('ERROR: Validation for the output matrix failed. Please check your code again.')
    
        
if __name__ == '__main__':
    """
    Generating a graph by R-MAT.
    optional arguments:
        -n NUMNODES, --numNodes NUMNODES Select the number of nodes of the graph.
        -e NUMEDGES, --numEdges NUMEDGES Select the number of edges of the graph.
        -a PA, --pA PA The probability of the edge assignment in the partition a.
        -b PB, --pB PB The probability of the edge assignment in the partition b.
        -c PC, --pC PC The probability of the edge assignment in the partition c.
        -o OUTPUTNAME, --output OUTPUTNAME The file name of an output matrix.

    Example:
        python main.py -n 128 -e 1000 -a 0.5 -b 0.2 -c 0.1 -o example.npy
    """
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Generating a graph by R-MAT.')
    parser.add_argument("-n", "--numNodes", metavar="NUMNODES", type=int, default=128,
                            help="Select the number of nodes of the graph.")
    parser.add_argument("-e", "--numEdges", metavar="NUMEDGES", type=int, default=1000,
                            help="Select the number of edges of the graph.")
    parser.add_argument("-a", "--pA", metavar="PA", type=float, default=0.25,
                            help="The probability of the edge assignment in the partition a.")
    parser.add_argument("-b", "--pB", metavar="PB", type=float, default=0.25,
                            help="The probability of the edge assignment in the partition b.")
    parser.add_argument("-c", "--pC", metavar="PC", type=float, default=0.25,
                            help="The probability of the edge assignment in the partition c.")
    parser.add_argument("-o", "--output", metavar="OUTPUT", type=str, default="example.npy",
                            help="The file name of an output matrix.")
    args = parser.parse_args()
    
    # Validation for inputs. DO *NOT* MODIFY THIS PART!
    valid = True
    if type(args.numNodes) != int or args.numNodes <= 0:
        print('ERROR: numNodes should be non-negative.')
        valid = False
    if type(args.numEdges) != int or args.numEdges <= 0 or args.numEdges > (args.numNodes * args.numNodes):
        print('ERROR: numEdges should be non-negative, and should not be exceed numNodes^2.')
        valid = False
    if type(args.pA) != float or args.pA < 0 or args.pA > 1:
        print('ERROR: Probabilites should be non-negative, and should not exceed 1.')
        valid = False
    if type(args.pB) != float or args.pB < 0 or args.pB > 1:
        print('ERROR: Probabilites should be non-negative, and should not exceed 1.')
        valid = False
    if type(args.pC) != float or args.pC < 0 or args.pC > 1:
        print('ERROR: Probabilites should be non-negative, and should not exceed 1.')
        valid = False
    if type(args.pA) != float or type(args.pB) != float or type(args.pC) != float or not (0 <= (args.pA + args.pB + args.pC) <= 1):
        print('ERROR: Probabilites should be non-negative, and should not exceed 1.')
        valid = False
    if type(args.output) != str or len(args.output) == 0:
        print('ERROR: You should specify the path of an output matrix.')
        valid = False
    if not valid:
        print('ERROR: Validation for inputs failed. Please check your code again.')
        exit(0)
        
    N, E = args.numNodes, args.numEdges
    probs = [args.pA, args.pB, args.pC, 1 - (args.pA + args.pB + args.pC)]
    output_path = args.output
    genRMat(N, E, probs, output_path)
    end_time = time.time()

    print(f'Total Time Spent: {(end_time-start_time)/60} minutes.')

    # python main.py -n 256 -e 2000 -a 0.25 -b 0.25 -c 0.25 -o G_1S
    # python main.py -n 256 -e 2000 -a 0.40 -b 0.20 -c 0.20 -o G_2S
    # python main.py -n 256 -e 2000 -a 0.70 -b 0.10 -c 0.10 -o G_3S
    # python main.py -n 16384 -e 200000 -a 0.25 -b 0.25 -c 0.25 -o G_1L
    # python main.py -n 16384 -e 200000 -a 0.40 -b 0.20 -c 0.20 -o G_2L
    # python main.py -n 16384 -e 200000 -a 0.70 -b 0.10 -c 0.10 -o G_3L