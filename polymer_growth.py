"""approximate partition functions via polymer growth"""

from __future__ import print_function

import networkx as nx
import numpy as np

np.seterr(all='warn')
np.seterr(under='ignore', over='ignore')


def safe_log(x):
    """need to guard for infs/zeros in logs here"""

    if x == 0.0:
        logx = -np.inf
    else:
        logx = np.log(x)

    return logx

# pylint: disable=too-many-locals
def get_betaU_for_state(some_graph, global_state):
    """
    return the energy of a global configuration of the graph
    need to pass the inverse node ordering map to get the correct position
    of the node in the global state when the nodes are added in random order
    """

    all_node_pots = nx.get_node_attributes(some_graph, 'node_potential')
    all_edge_pots = nx.get_edge_attributes(some_graph, 'edge_potential')

    betaU_state_node = 0.0
    for node in some_graph.nodes_iter():
        # print('node = ', node)
        # print('global_state = ', global_state)
        node_state = global_state[node]
        # 0 cause col vector is 2-dim:
        phi_n = all_node_pots[node][node_state, 0]
        betaU_state_node += -safe_log(phi_n)

    betaU_state_edge = 0.0
    for edge in some_graph.edges_iter():
        # print('')
        # print('getting betaU for edge', edge)
        # print('global_state =', global_state)
        n_a, n_b = edge
        # print('edge_pot =', all_edge_pots[edge])
        phi_e = all_edge_pots[edge][global_state[n_a]][global_state[n_b]]

        betaU_state_edge += -safe_log(phi_e)

    betaU_state = (betaU_state_node + betaU_state_edge)

    return betaU_state
# pylint: enable=too-many-locals


def boltzman_sample_state_ens(state_list, betaE_list, sample_size=10):
    """
    boltzman sampling of states. betaE_list is used to weight the samples
    """

    state_inds = [i for (i, _) in enumerate(state_list)]
    finite_inds = [i for i in state_inds if betaE_list[i] < np.inf and betaE_list[i] > -np.inf]
    assert len(finite_inds) > 0, 'no states have finite energy!'

    finite_states = [state_list[i] for i in finite_inds]
    finite_betaE_list = [betaE_list[i] for i in finite_inds]
    if np.any(~np.isfinite(finite_betaE_list)):
        print(finite_betaE_list)

    finite_beta_Emin = np.min(finite_betaE_list)
    finite_state_probs = np.exp(-(np.array(finite_betaE_list) - finite_beta_Emin))
    finite_state_probs /= np.sum(finite_state_probs)

    finite_inds_sample = np.random.choice(range(len(finite_states)), size=sample_size, replace=True, p=finite_state_probs)
    state_sample = [finite_states[i] for i in finite_inds_sample]

    return state_sample


def calc_betaF(betaEs, sample_size, phys=True):
    """
    safer way to compute the free energy:

    betaF   = -np.log(np.sum(np.exp(-betaEs))/sample_size)
            = -np.log(np.exp(-betaEmin)*np.sum(np.exp(-(betaEs-betaEmin)))/sample_size)
            = -np.log(np.exp(-betaEmin)) - np.log(np.sum(np.exp(-(betaEs-betaEmin)))) + np.log(sample_size)
            = betaEmin - np.log(np.sum(np.exp(-(betaEs-betaEmin)))) + np.log(sample_size)
    """

    betaEs = np.array(betaEs)
    betaEmin = np.min(betaEs)
    betaF = betaEmin - np.log(np.sum(np.exp(-(betaEs - betaEmin)))) + np.log(sample_size)
    if phys:
        pass

    return betaF


# pylint: disable=too-many-locals
def polymer_growth(graph, sample_size=10, shuffle=False, phys=False):
    """
    polymer growth sampling of a protein/peptide graph
    inputs:
        g: graph / markov random field of peptide/protein
        sample_size: number of states we downsample to as each residue is added
        shuffle: reorder nodes, or not
        phys: correct for phase space volumes or not
    outputs:
        betaF: list of delta betaF values for adding each successive nodes (and it's edges) to the graph
        betaU: list of delta betaU values for adding each successive nodes (and it's edges) to the graph
        betaTS: list of delta betaTS values for adding each successive nodes (and it's edges) to the graph
        note: if phys = true, the phys correction is appended to these lists
    """

    # get the node indices
    original_nodes = np.array([n for n in graph.nodes_iter()])

    # add the nodes in a random order or not
    if shuffle:
        node_order = list(np.random.permutation(original_nodes))
    else:
        node_order = original_nodes

    print('shuffled node order =', node_order)

    # index the nodes from zero
    node_mapping = dict(zip(node_order, xrange(len(node_order))))
    inv_node_mapping = {v: k for k, v in node_mapping.items()}


    # create copy of graph with nodes renamed as integers
    g = nx.relabel_nodes(graph, node_mapping)

    # if the nodes get relabeld out of order, transpose edge pot so lower index is the row
    for m, n in g.edges():
        if inv_node_mapping[m] > inv_node_mapping[n]:
            g.edge[n][m]['edge_potential'] = g.edge[n][m]['edge_potential'].transpose()

    print('integer nodes = ', g.nodes())

    # print('')
    # print('node potentials:')
    # print(nx.get_node_attributes(g, 'node_potential'))
    #
    # print('')
    # print('edge potentials:')
    # print(nx.get_edge_attributes(g, 'edge_potential'))

    # array of delta beta F/U/TS values
    delta_betaF = np.zeros(len(g.nodes()))

    # initialize 'polymer' -- each state is a list, saved_states is a list of lists
    saved_states = [[]]
    saved_ens = [0.0]

    # successively add each node, calculate delta F, and downsample the product state-space
    nodes = [n for n in g.nodes_iter()]
    for node in nodes:
        print('added node', node)

        # h is a temp graph with nodes/edges > i removed
        h = g.copy()
        h.remove_nodes_from(nodes[node + 1:])
        print('h nodes = ', h.nodes())


        new_states = []
        delta_ens = []

        # check all the new states we get by adding in a new node to the old states
        for node_state in h.node[node]['node_state_indices']:
            for i, old_state in enumerate(saved_states):
                new_state = old_state + [node_state]
                new_states.append(new_state)
                new_en = get_betaU_for_state(h, new_state)

                delta_en = new_en - saved_ens[i]
                delta_ens.append(delta_en)

        delta_betaF[node] = calc_betaF(delta_ens, len(saved_states))
        saved_states = boltzman_sample_state_ens(new_states, delta_ens, sample_size=sample_size)
        saved_ens = [get_betaU_for_state(h, state) for state in saved_states]

    # after all the nodes are added and the graph is rebuilt, compute our stats:
    betaF = np.sum(delta_betaF)
    betaU = np.mean([get_betaU_for_state(h, state) for state in saved_states])
    betaTS = betaU - betaF

    if phys:
        d_c, d_h = g.graph['num_chis']
        k_c, k_h = g.graph['grid_points_per_chi']
        discretization_correction = d_c * np.log(k_c / (2 * np.pi)) + d_h * np.log(k_h / (2 * np.pi / 3))
        betaF += discretization_correction
        betaTS -= discretization_correction

    return betaF, betaU, betaTS
# pylint: enable=too-many-locals
