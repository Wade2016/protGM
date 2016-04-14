"""
Construct test MRFs, some static and some dynamic/random
"""

# construct some specific graphs for tests:

# - 3-node chain, with set potentials
# - 3-node loop from chris's paper
# - N-node chain with random potentials
# - N-node loop with random potentials
# - N-node Ising chain
# - N-node Ising loop

# - N-node random graph?

import numpy as np
import networkx as nx


def make_test_graph():
    """
    make graph: n_0---n_1---n_2 ; |n_0| = 4, |n_1| = 5, |n_2| = 2
    """

    node_potentials = {}
    edge_potentials = {}

    # pylint: disable=line-too-long
    node_potentials[0] = np.array([[1.93076941], [0.67489273], [1.29234289], [1.19193917]])
    node_potentials[1] = np.array([[1.10842027], [0.54192028], [0.53596490], [0.14470498], [0.57183493]])
    node_potentials[2] = np.array([[1.33507577], [0.34522988]])

    edge_potentials[(0, 1)] = np.array([[1.21435952, 0.78865583, 5.93958029, 0.91751033, 3.57118778],
                                        [1.25370960, 0.68910615, 1.69770695, 0.93913777, 0.93399517],
                                        [0.79529185, 0.20405205, 0.38127442, 0.08784021, 2.05977868],
                                        [0.86819407, 0.17969205, 1.25121786, 2.52078906, 1.08530778]])
    edge_potentials[(1, 2)] = np.array([[0.50935666, 0.25665769],
                                        [0.76283116, 2.39419747],
                                        [4.17448190, 1.07067697],
                                        [0.30792333, 0.54189997],
                                        [0.50144309, 0.19648555]])
    # pylint: enable=line-too-long

    # initialize and add nodes
    graph_name = nx.Graph()
    graph_name.add_nodes_from(xrange(len(node_potentials)))

    # add edges
    for node in xrange(1, len(graph_name.nodes())):
        graph_name.add_edge(node - 1, node)

    # add potentials to graph
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_potential'] = node_potentials[n]
    for edge in graph_name.edges_iter():
        n_a, n_b = edge
        graph_name.edge[n_a][n_b]['edge_potential'] = edge_potentials[edge]

    # add indices of node state -- seems like there should be a slick way to
    # avoid this but can't figure it out
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_state_indices'] = [
            i for i in xrange(len(graph_name.node[n]['node_potential']))]

    return graph_name


def make_chris_graph():
    """
    make graph from the example at the end of hetu and chris's paper:
    http://www.ncbi.nlm.nih.gov/pubmed/21120864
    the graph has three nodes and three edges
    """
    node_potentials = {}
    node_potentials[0] = np.array([[2.0], [2.0]])
    node_potentials[1] = np.array([[1.5], [2.5]])
    node_potentials[2] = np.array([[1.0], [1.0]])

    edge_potentials = {}
    edge_potentials[(0, 1)] = np.array([[0.1, 0.4],
                                        [0.1, 0.9]])
    edge_potentials[(1, 2)] = np.array([[0.1, 1.4],
                                        [0.5, 0.1]])
    edge_potentials[(0, 2)] = np.array([[1.3, 0.2],
                                        [0.5, 0.7]])

    # initialize and add nodes
    graph_name = nx.Graph()
    graph_name.add_nodes_from(xrange(len(node_potentials)))

    # add edges
    for node in xrange(1, len(graph_name.nodes())):
        graph_name.add_edge(node - 1, node)
    graph_name.add_edge(0, 2)

    # add potentials to graph
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_potential'] = node_potentials[n]
    for edge in graph_name.edges_iter():
        n_a, n_b = edge
        graph_name.edge[n_a][n_b]['edge_potential'] = edge_potentials[edge]

    # add indices of node state -- seems like there should be a slick way to
    # avoid this but can't figure it out
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_state_indices'] = [
            i for i in xrange(len(graph_name.node[n]['node_potential']))]

    return graph_name


def make_random_chain_or_loop_graph(N, loop):
    """
    make random chain graph: n_0---n_1---...---n_N;
    states per node vary from 2 to 5
    loop has Nth node connected to 0th node; chain doesn't
    """
    N_nodes = N
    node_state_sizes = np.random.randint(2, high=50, size=N_nodes)

    # initialize and add nodes
    graph_name = nx.Graph()
    graph_name.add_nodes_from(xrange(N_nodes))

    # add indices of node state -- seems like there should be a slich way to
    # avoid this but can't figure it out
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_state_indices'] = [
            i for i in xrange(node_state_sizes[n])]

    # add edges
    for node in xrange(1, len(graph_name.nodes())):
        graph_name.add_edge(node - 1, node)

    if loop:
        graph_name.add_edge(0, N - 1)

    # add potentials to graph
    for n in graph_name.nodes_iter():
        node_potential = np.random.exponential(1, [node_state_sizes[n], 1])
        graph_name.node[n]['node_potential'] = node_potential
    for i, edge in enumerate(graph_name.edges_iter()):
        n_a, n_b = edge
        len_n_a = len(graph_name.node[n_a]['node_potential'])
        len_n_b = len(graph_name.node[n_b]['node_potential'])
        edge_potential = np.random.exponential(1, [len_n_a, len_n_b])
        graph_name.edge[n_a][n_b]['edge_potential'] = edge_potential

    return graph_name


def make_random_chain_graph(N):
    """
    make random chain graph: n_0---n_1---...---n_N;
    states per node vary from 2 to 5
    """
    return make_random_chain_or_loop_graph(N, False)


def make_random_loop_graph(N):
    """
                             +-----------------+
                             |                 |
    make random loop graph: n_0---n_1---...---n_N;
    states per node vary from 2 to 5
    """
    return make_random_chain_or_loop_graph(N, True)


def make_ising_chain_or_loop_graph(N, loop, H=1.0, J=1.0):
    """
    make graph of N-node ising loop or chain
    loop has Nth node connected to 0th node; chain doesn't
    """

    node_energy = np.array([[-H], [H]])
    node_potential = np.exp(-1 * node_energy)

    edge_energy = np.array([[-J, J], [J, -J]])
    edge_potential = np.exp(-1 * edge_energy)

    # initialize and add nodes
    graph_name = nx.Graph()
    graph_name.add_nodes_from(xrange(N))

    # add edges
    for node in xrange(1, len(graph_name.nodes())):
        graph_name.add_edge(node - 1, node)

    # add that last edge if we want a loop
    if loop:
        graph_name.add_edge(0, N - 1)

    # add potentials to graph
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_potential'] = node_potential
    for i, edge in enumerate(graph_name.edges_iter()):
        n_a, n_b = edge
        graph_name.edge[n_a][n_b]['edge_potential'] = edge_potential

    # add indices of node state -- seems like there should be a slick way to
    # avoid this but can't figure it out
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_state_indices'] = [
            i for i in xrange(len(graph_name.node[n]['node_potential']))]

    return graph_name


def make_ising_chain_graph(N, H=1.0, J=1.0):
    """make graph of N-node ising chain"""
    return make_ising_chain_or_loop_graph(N, False, H, J)


def make_ising_loop_graph(N, H=1.0, J=1.0):
    """make graph of N-node ising loop"""
    return make_ising_chain_or_loop_graph(N, True, H, J)


def make_random_binomial_graph(N_nodes, p_edge):
    """
    random binomially connected graph with exponentially distributed potentials
    """

    # initialize and add nodes
    graph_name = nx.binomial_graph(N_nodes, p_edge)
    node_state_sizes = np.random.randint(
        10, high=20, size=graph_name.number_of_nodes())

    # add indices of node state -- seems like there should be a slick way to
    # avoid this but can't figure it out
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_state_indices'] = [
            i for i in xrange(node_state_sizes[n])]

    # add potentials to graph
    for n in graph_name.nodes_iter():
        node_potential = np.random.exponential(1, [node_state_sizes[n], 1])
        graph_name.node[n]['node_potential'] = node_potential
    for i, edge in enumerate(graph_name.edges_iter()):
        n_a, n_b = edge

        len_n_a = len(graph_name.node[n_a]['node_potential'])
        len_n_b = len(graph_name.node[n_b]['node_potential'])
        edge_potential = np.random.exponential(1, [len_n_a, len_n_b])
        graph_name.edge[n_a][n_b]['edge_potential'] = edge_potential

    return graph_name


def make_random_tree_graph(N_nodes, gamma_powerlaw=3, N_tries=1000):
    """random tree graph with exponentially distributed potentials"""
    # initialize and add nodes
    graph_name = nx.random_powerlaw_tree(
        N_nodes, gamma=gamma_powerlaw, tries=N_tries)
    node_state_sizes = np.random.randint(
        2, high=5, size=graph_name.number_of_nodes())

    # add indices of node state -- seems like there should be a slick way to
    # avoid this but can't figure it out
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_state_indices'] = [
            i for i in xrange(node_state_sizes[n])]

    # add potentials to graph
    for n in graph_name.nodes_iter():
        node_potential = np.random.exponential(1, [node_state_sizes[n], 1])
        graph_name.node[n]['node_potential'] = node_potential
    for i, edge in enumerate(graph_name.edges_iter()):
        n_a, n_b = edge
        len_n_a = len(graph_name.node[n_a]['node_potential'])
        len_n_b = len(graph_name.node[n_b]['node_potential'])
        edge_potential = np.random.exponential(1, [len_n_a, len_n_b])
        graph_name.edge[n_a][n_b]['edge_potential'] = edge_potential

    return graph_name


def make_random_lobster(N_nodes, p1=0.5, p2=0.5):
    """random lobster graph with exponentially distributed potentials"""

    # initialize and add nodes
    graph_name = nx.random_lobster(N_nodes, p1, p2, seed=None)
    node_state_sizes = np.random.randint(
        2, high=5, size=graph_name.number_of_nodes())

    # add indices of node state -- seems like there should be a slick way to
    # avoid this but can't figure it out
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_state_indices'] = [
            i for i in xrange(node_state_sizes[n])]

    # add potentials to graph
    for n in graph_name.nodes_iter():
        node_potential = np.random.exponential(1, [node_state_sizes[n], 1])
        graph_name.node[n]['node_potential'] = node_potential
    for i, edge in enumerate(graph_name.edges_iter()):
        n_a, n_b = edge
        len_n_a = len(graph_name.node[n_a]['node_potential'])
        len_n_b = len(graph_name.node[n_b]['node_potential'])
        edge_potential = np.random.exponential(1, [len_n_a, len_n_b])
        graph_name.edge[n_a][n_b]['edge_potential'] = edge_potential

    return graph_name
