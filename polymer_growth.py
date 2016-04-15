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
def get_betaU_for_state(some_graph, global_state, inv_node_map):
    """
    return the energy of a global configuration of the graph
    need to pass the inverse node ordering map to get the correct position
    of the node in the global state when the nodes are added in random order
    """

    all_node_pots = nx.get_node_attributes(some_graph, 'node_potential')
    all_edge_pots = nx.get_edge_attributes(some_graph, 'edge_potential')

    betaU_state_node = 0.0
    for node in some_graph.nodes_iter():
        k = inv_node_map[node]
        node_state = global_state[k]
        # 0 cause col vector is 2-dim:
        phi_n = all_node_pots[node][node_state, 0]
        betaU_state_node += -safe_log(phi_n)

    betaU_state_edge = 0.0
    for edge in some_graph.edges_iter():
        n_a, n_b = edge
        i = inv_node_map[n_a]
        j = inv_node_map[n_b]
        phi_e = all_edge_pots[edge][global_state[i]][global_state[j]]
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
def polymer_growth(g, sample_size=10, shuffle=False, phys=False):
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
    nodes = np.array([n for n in g.nodes_iter()])
    all_node_state_indices = nx.get_node_attributes(g, 'node_state_indices')

    # add the nodes in a random order or not
    if shuffle:
        node_order = np.random.permutation(nodes)
    else:
        node_order = nodes

    # node_map = dict(zip(nodes, node_order))
    node_map_back = dict(zip(node_order, nodes))

    all_node_state_indices_ordered = [all_node_state_indices[k] for k in node_order]

    # array of delta beta F/U/TS values
    delta_betaF = np.zeros(len(nodes))

    # initialize 'polymer' -- each state is a list, saved_states is a list of lists
    saved_states = [[]]
    saved_ens = [0.0]

    # successively add each node, calculate delta F, and downsample the product state-space
    for k, n in enumerate(node_order):

        # h is a temp graph with nodes/edges > i removed
        h = g.copy()
        h.remove_nodes_from(node_order[k + 1:])

        new_states = []
        delta_ens = []

        # check all the new states we get by adding in a new node to the old states
        for node_state in all_node_state_indices_ordered[k]:
            for i, old_state in enumerate(saved_states):
                new_state = old_state + [node_state]
                new_states.append(new_state)
                new_en = get_betaU_for_state(h, new_state, node_map_back)

                delta_en = new_en - saved_ens[i]
                delta_ens.append(delta_en)

        delta_betaF[k] = calc_betaF(delta_ens, len(saved_states))
        saved_states = boltzman_sample_state_ens(
            new_states, delta_ens, sample_size=sample_size)
        saved_ens = [get_betaU_for_state(
            h, state, node_map_back) for state in saved_states]

    # after all the nodes are added and the graph is rebuilt, compute our stats:
    betaF = np.sum(delta_betaF)
    betaU = np.mean([get_betaU_for_state(h, state, node_map_back) for state in saved_states])
    betaTS = betaU - betaF

    if phys:
        d_c, d_h = g.graph['num_chis']
        k_c, k_h = g.graph['grid_points_per_chi']
        discretization_correction = d_c * np.log(k_c / (2 * np.pi)) + d_h * np.log(k_h / (2 * np.pi / 3))
        betaF += discretization_correction
        betaTS -= discretization_correction

    return betaF, betaU, betaTS
# pylint: enable=too-many-locals
