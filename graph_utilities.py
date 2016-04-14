"""
Utility functions for inspecting and drawing graphs
"""

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def inspect_graph(graph_name):
    """
    prints some MRF info
    """
    print "nodes", nx.nodes(graph_name)
    print "edges", nx.edges(graph_name)

    for node in graph_name.nodes_iter():
        print "node {0} indices = ".format(node), graph_name.node[node]['node_state_indices']

    for node in graph_name.nodes_iter():
        print "phi_{0} = ".format(node), graph_name.node[node]['node_potential']

    for edge in graph_name.edges_iter():
        node_a, node_b = edge
        print "phi_{0}{1} = ".format(node_a, node_b)
        print graph_name.edge[node_a][node_b]['edge_potential']


def draw_graph(graph, graph_layout='shell',
               node_size=1600, node_color='lightgray', node_alpha=1.0,
               node_text_size=12,
               edge_color='lightgray', edge_alpha=1.0, edge_tickness=2,
               text_font='sans-serif'):
    """draws a graph using networkx with some sane defaults.  pretty useless
    when the number of nodes is greater than about 50-100"""

    if graph_layout == 'spring':
        graph_pos = nx.spring_layout(graph)
    elif graph_layout == 'spectral':
        graph_pos = nx.spectral_layout(graph)
    elif graph_layout == 'random':
        graph_pos = nx.random_layout(graph)
    else:
        graph_pos = nx.shell_layout(graph)

    nx.draw_networkx_nodes(graph, graph_pos, node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(graph, graph_pos, width=edge_tickness,
                           alpha=edge_alpha, edge_color=edge_color)
    nx.draw_networkx_labels(graph, graph_pos, font_size=node_text_size,
                            font_family=text_font)
    sns.set_style("white")
    plt.axis('off')
    plt.show()
