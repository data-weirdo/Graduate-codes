import numpy as np
import random
### TODO: only importing INTERNAL libraries are allowed except for NumPy ###
from itertools import compress
from time import time
############################################################################

def sampleInitialNodes(num_nodes, num_initial_nodes):
    """ 
    Implement the function that samples the initial active nodes.
    
    Inputs:
        num_nodes: the number of nodes N in the given graph. 
        num_initial_nodes: the target number of initial active nodes N_0.
        
    Output: a list of N_0 initial active nodes
    """
    ### TODO: WRITE YOUR CODE HERE ###
    candidate = list(range(num_nodes))
    random.shuffle(candidate)
    initial = candidate[:num_initial_nodes]

    return initial
    ##################################




def runSimulation(num_nodes, edges, initial_nodes, beta, delta):
    """ 
    Implement the function that counts the ultimate number of recovered nodes.
    
    Inputs:
        num_nodes: the number of nodes in the input graph. 
        edges: a list of tuples of the form (u, v),
          which indicates that there is an edge from u to v. 
          You can assume that there is no isolated node.
        initial_nodes: a list of initial active nodes.
        beta: the infection rate of the SIR model
        delta: the recovery rate of the SIR model
        
    Output: the number of recovered nodes after the diffusion process ends.
    """
    ### TODO: WRITE YOUR CODE HERE ###

    # NOTE Only deals with susceptible-infected cases.
    def _get_infected(susceptible, infected, edge_list, beta):
        possible_interaction = [x for x in edge_list if x[0] in infected] # x[0]: src node
        infected_or_not = np.random.binomial(1, beta, len(possible_interaction)).tolist()

        edges_of_infection = list(compress(possible_interaction, infected_or_not))
        to_infected = list(set([x[1] for x in edges_of_infection])) # x[1]: dst node

        still_susceptible = list(set(susceptible) - set(to_infected))

        edge_list = [x for x in edge_list if x[1] not in to_infected]

        return still_susceptible, to_infected, edge_list

    # NOTE Only deals with infected-recovered cases
    def _get_recovered(infected, recovered, edge_list, delta):
        recovered_or_not = np.random.binomial(1, delta, len(infected)).tolist()

        to_recovered = list(compress(infected, recovered_or_not))
        edge_list = [instance for instance in edge_list if instance[0] not in to_recovered]
        edge_list = [instance for instance in edge_list if instance[1] not in to_recovered]

        still_infected = list(set(infected) - set(to_recovered))
        recovered.extend(to_recovered)

        return still_infected, recovered, edge_list

    # At time 0 (Initial step)
    susceptible = list(set(list(range(num_nodes))) - set(initial_nodes))
    infected = initial_nodes
    recovered = []
    edges = [x for x in edges if x[1] not in infected]

    # Actual implementation 
    while len(infected) != 0:
        susceptible, newly_infected, edges = _get_infected(susceptible, infected, edges, beta)
        old_infected, recovered, edges = _get_recovered(infected, recovered, edges, delta)
        infected = newly_infected + old_infected
        assert len(recovered) == len(list(set(recovered))), 'There might be problem.'

    return len(recovered)
    #################################




# def runSimulation(num_nodes, edges, initial_nodes, beta, delta):
#     """ 
#     Implement the function that counts the ultimate number of recovered nodes.
    
#     Inputs:
#         num_nodes: the number of nodes in the input graph. 
#         edges: a list of tuples of the form (u, v),
#           which indicates that there is an edge from u to v. 
#           You can assume that there is no isolated node.
#         initial_nodes: a list of initial active nodes.
#         beta: the infection rate of the SIR model
#         delta: the recovery rate of the SIR model
        
#     Output: the number of recovered nodes after the diffusion process ends.
#     """
#     ### TODO: WRITE YOUR CODE HERE ###

#     # NOTE Only deals with susceptible-infected cases.
#     def _get_infected(susceptible, infected, edge_list, beta):
#         possible_interaction = [x for x in edge_list if x[0] in infected] # x[0]: src node
#         infected_or_not = np.random.binomial(1, beta, len(possible_interaction)).tolist()

#         edges_of_infection = list(compress(possible_interaction, infected_or_not))
#         to_infected = list(set([x[1] for x in edges_of_infection])) # x[1]: dst node

#         still_susceptible = list(set(susceptible) - set(to_infected))

#         edge_list = [x for x in edge_list if x[1] not in to_infected]

#         return still_susceptible, to_infected, edge_list

#     # NOTE Only deals with infected-recovered cases
#     def _get_recovered(infected, recovered, edge_list, delta):
#         recovered_or_not = np.random.binomial(1, delta, len(infected)).tolist()

#         to_recovered = list(compress(infected, recovered_or_not))
#         edge_list = [instance for instance in edge_list if instance[0] not in to_recovered]
#         edge_list = [instance for instance in edge_list if instance[1] not in to_recovered]

#         still_infected = list(set(infected) - set(to_recovered))
#         recovered.extend(to_recovered)

#         return still_infected, recovered, edge_list

#     # At time 0 (Initial step)
#     susceptible = list(set(list(range(num_nodes))) - set(initial_nodes))
#     infected = initial_nodes
#     recovered = []
#     edges = [x for x in edges if x[1] not in infected]

#     # Actual implementation 
#     while len(infected) != 0:
#         susceptible, newly_infected, edges = _get_infected(susceptible, infected, edges, beta)
#         old_infected, recovered, edges = _get_recovered(infected, recovered, edges, delta)
#         infected = newly_infected + old_infected
#         assert len(recovered) == len(list(set(recovered))), 'There might be problem.'

#     return len(recovered)
#     ###############################