"""
exact partition function computations for specific graph structures as tests
and arbitrary graphs for production
"""

import itertools as it
import networkx as nx
import numpy as np


def clean_up_graph(g, pot_min=0.0, pot_max=1.0e+100):
    """clip graph potentials to remove infs"""
    for edge in g.edges_iter():
        n, m = edge
        edge_pot = g.edge[n][m]['edge_potential']
        edge_pot = np.clip(edge_pot, pot_min, pot_max)
        g.edge[n][m]['edge_potential'] = edge_pot

    for node in g.nodes_iter():
        node_pot = g.node[node]['node_potential']
        node_pot = np.clip(node_pot, pot_min, pot_max)
        g.node[node]['node_potential'] = node_pot


def exact_pfunc_three_chain(graph_name, debug=False):
    """gives the exact partition function for a 3-node chain graph"""
    Z = 0
    for i in graph_name.node[0]['node_state_indices']:
        for j in graph_name.node[1]['node_state_indices']:
            for k in graph_name.node[2]['node_state_indices']:
                z_node = graph_name.node[0]['node_potential'][i] \
                    * graph_name.node[1]['node_potential'][j] \
                    * graph_name.node[2]['node_potential'][k]
                z_edge = graph_name.edge[0][1]['edge_potential'][i][j] \
                    * graph_name.edge[1][2]['edge_potential'][j][k]
                z = z_node * z_edge
                Z += z
                if debug:
                    print "state = ({0}, {1}, {2}):".format(i, j, k)
                    print "  ", "phi_0({0}) = {1}".format(i, graph_name.node[0]['node_potential'][i]), "phi_1({0}) = {1}".format(j, graph_name.node[1]['node_potential'][j]), "phi_2({0}) = {1}".format(k, graph_name.node[2]['node_potential'][k])
                    print "  ", "phi_01({0},{1}) = {2}".format(i, j, graph_name.edge[0][1]['edge_potential'][i][j]), "phi_12({0},{1}) = {2}".format(j, k, graph_name.edge[1][2]['edge_potential'][j][k])
                    print "  ", "z_node = {0}".format(z_node)
                    print "  ", "z_edge = {0}".format(z_edge)
                    print "  ", "z = {0}".format(z)
    return Z


def exact_pfunc_three_loop(graph_name, debug=False):
    """gives the exact partition function for a 3-node loop graph"""
    Z = 0
    for i in graph_name.node[0]['node_state_indices']:
        for j in graph_name.node[1]['node_state_indices']:
            for k in graph_name.node[2]['node_state_indices']:
                z_node = graph_name.node[0]['node_potential'][i] \
                    * graph_name.node[1]['node_potential'][j] \
                    * graph_name.node[2]['node_potential'][k]
                z_edge = graph_name.edge[0][1]['edge_potential'][i][j] \
                    * graph_name.edge[1][2]['edge_potential'][j][k] \
                    * graph_name.edge[0][2]['edge_potential'][i][k]
                z = z_node * z_edge
                Z += z
                if debug:
                    print "state = ({0}, {1}, {2}):".format(i, j, k)
                    print "  ", "phi_0({0}) = {1}".format(i, graph_name.node[0]['node_potential'][i]), "phi_1({0}) = {1}".format(j, graph_name.node[1]['node_potential'][j]), "phi_2({0}) = {1}".format(k, graph_name.node[2]['node_potential'][k])
                    print "  ", "phi_01({0},{1}) = {2}".format(i, j, graph_name.edge[0][1]['edge_potential'][i][j]), "phi_02({0},{1}) = {2}".format(i, k, graph_name.edge[0][2]['edge_potential'][i][k]), "phi_12({0},{1}) = {2}".format(j, k, graph_name.edge[1][2]['edge_potential'][j][k])
                    print "  ", "z_node = {0}".format(z_node)
                    print "  ", "z_edge = {0}".format(z_edge)
                    print "  ", "z = {0}".format(z)
    return Z


def ising_3_loop_pfunc_from_formula(H=1.0, J=1.0):
    """exact formula for 3-node ising loop"""
    Z = np.exp(3 * (H + J)) + 3 * np.exp(H - J) + 3 * \
        np.exp(-H - J) + np.exp(3 * (-H + J))
    return Z

# pylint: disable=too-many-locals
def exact_pfunc(some_graph, phys=False):
    """
    exact partition function for any graph.
    global state is a tuple of state indices for all nodes in the graph
    eg, for a three node graph, global_state = (4,2,1) implies the
    0th node in 4th state, 1st node in 2nd state, third node in 1st state.
    nodes do not need to have the same number of possible states.
    """

    clean_up_graph(some_graph)

    all_var_inds = nx.get_node_attributes(some_graph, 'node_state_indices')
    all_node_pots = nx.get_node_attributes(some_graph, 'node_potential')
    all_edge_pots = nx.get_edge_attributes(some_graph, 'edge_potential')

    Z = 0.0
    for global_state in it.product(*all_var_inds.itervalues()):
        z = 0.0

        z_node = 1.0
        for node in some_graph.nodes_iter():
            # 0 cause col vector is 2-dim
            z_node *= all_node_pots[node][global_state[node], 0]

        z_edge = 1.0
        for edge in some_graph.edges_iter():
            n_a, n_b = edge
            z_edge *= all_edge_pots[edge][global_state[n_a]][global_state[n_b]]

        z = z_node * z_edge
        Z += z

    # only works for symmetry=3 right now
    if phys:
        d_c, d_h = some_graph.graph['num_chis']
        k_c, k_h = some_graph.graph['grid_points_per_chi']
        discretization_correction = ((2 * np.pi / k_c)**d_c) * (((2 * np.pi / 3) / k_h)**d_h)
        Z *= discretization_correction
    return Z
# pylint: enable=too-many-locals


def get_state_probs(some_graph):
    """
    inefficient way of calculating state probabilities by recalculating Z.
    """
    clean_up_graph(some_graph)

    all_var_inds = nx.get_node_attributes(some_graph, 'node_state_indices')
    all_node_pots = nx.get_node_attributes(some_graph, 'node_potential')
    all_edge_pots = nx.get_edge_attributes(some_graph, 'edge_potential')

    state_probs = {}

    Z = exact_pfunc(some_graph)

    for global_state in it.product(*all_var_inds.itervalues()):
        z = 0.0

        z_node = 1.0
        for node in some_graph.nodes_iter():
            # 0 cause col vector is 2-dim
            z_node *= all_node_pots[node][global_state[node], 0]

        z_edge = 1.0
        for edge in some_graph.edges_iter():
            n_a, n_b = edge
            z_edge *= all_edge_pots[edge][global_state[n_a]][global_state[n_b]]

        z = z_node * z_edge / Z

        state_probs[global_state] = z

    return state_probs

# pylint: disable=too-many-locals
def exact_betaU(some_graph, verbose=False):
    """
    exact average potential energy of graph, via <E>
    """
    clean_up_graph(some_graph)

    all_var_inds = nx.get_node_attributes(some_graph, 'node_state_indices')
    all_node_pots = nx.get_node_attributes(some_graph, 'node_potential')
    all_edge_pots = nx.get_edge_attributes(some_graph, 'edge_potential')

    p = get_state_probs(some_graph)

    betaU_tot = 0.0
    for global_state in it.product(*all_var_inds.itervalues()):

        if p[global_state] > 0.0:

            betaU_state_node = 0.0
            for node in some_graph.nodes_iter():
                # 0 because col vector is 2-dim:
                phi_n = all_node_pots[node][global_state[node], 0]
                betaU_state_node += -np.log(phi_n)

            betaU_state_edge = 0.0
            for edge in some_graph.edges_iter():
                n_a, n_b = edge
                phi_e = all_edge_pots[edge][
                    global_state[n_a]][global_state[n_b]]
                betaU_state_edge += -np.log(phi_e)
            if verbose:
                print 'p[{0}] = '.format(global_state), p[global_state]
            betaU_state = (betaU_state_node +
                           betaU_state_edge) * p[global_state]
            if verbose:
                print 'betaU_state = ', betaU_state

            betaU_tot += betaU_state

    return betaU_tot
# pylint: enable=too-many-locals

def exact_betaTS_direct(some_graph, phys=False):
    """
    exact entropy via p*logp of state probs
    """
    clean_up_graph(some_graph)

    states_and_probs = get_state_probs(some_graph)
    p = np.array([sp for sp in states_and_probs.itervalues()])
    nz_inds = p > 0
    betaTS = -np.sum(p[nz_inds] * np.log(p[nz_inds]))
    # only works for symmetry=3 right now
    if phys:
        d_c, d_h = some_graph.graph['num_chis']
        k_c, k_h = some_graph.graph['grid_points_per_chi']
        discretization_correction = d_c * \
            np.log(k_c / (2 * np.pi)) + d_h * np.log(k_h / (2 * np.pi / 3))
        betaTS -= discretization_correction
    return betaTS


def exact_betaG(some_graph, phys=False):
    """
    exact free energy via logZ
    """
    clean_up_graph(some_graph)

    betaG = -np.log(exact_pfunc(some_graph, phys=phys))
    return betaG


def exact_betaTS(some_graph, phys=False):
    """
    exact entropy via energy - free energy
    """
    clean_up_graph(some_graph)

    betaG = exact_betaG(some_graph, phys=phys)
    betaU = exact_betaU(some_graph)
    betaTS = betaU - betaG
    return betaTS

# pylint: disable=too-many-locals
def random_pfunc(some_graph, nsamples=100, phys=False):
    """
    radnomly sampled partition function for any graph.
    global state is a tuple of state indices for all nodes in the graph
    eg, for a three node graph, global_state = (4,2,1) implies the
    0th node in 4th state, 1st node in 2nd state, third node in 1st state.
    nodes do not need to have the same number of possible states.
    """

    clean_up_graph(some_graph)

    all_var_inds = nx.get_node_attributes(some_graph, 'node_state_indices')
    all_node_pots = nx.get_node_attributes(some_graph, 'node_potential')
    all_edge_pots = nx.get_edge_attributes(some_graph, 'edge_potential')

    Z = 0.0

    globals_states = [[np.random.choice(all_var_inds[k]) for k in all_var_inds.keys()] for _ in xrange(nsamples)]
    for global_state in globals_states:
        z = 0.0

        z_node = 1.0
        for node in some_graph.nodes_iter():
            # 0 cause col vector is 2-dim
            z_node *= all_node_pots[node][global_state[node], 0]

        z_edge = 1.0
        for edge in some_graph.edges_iter():
            n_a, n_b = edge
            z_edge *= all_edge_pots[edge][global_state[n_a]][global_state[n_b]]

        z = z_node * z_edge
        Z += z

    if phys:
        Z /= float(nsamples)
    return Z
# pylint: enable=too-many-locals


def random_betaG(some_graph, nsamples=100, phys=False):
    """
    exact free energy via logZ
    """
    clean_up_graph(some_graph)

    betaG = -np.log(random_pfunc(some_graph, nsamples=nsamples, phys=phys))
    return betaG
