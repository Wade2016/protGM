import numpy as np
import networkx as nx
import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

## utility to inspect constructed graphs

def inspect_graph(graph_name):
    print "nodes", nx.nodes(graph_name)
    print "edges", nx.edges(graph_name)
    
    for n in graph_name.nodes_iter():
        print "node {0} indices = ".format(n), graph_name.node[n]['node_state_indices']
    
    for n in graph_name.nodes_iter():
        print "phi_{0} = ".format(n), graph_name.node[n]['node_potential']

    for edge in graph_name.edges_iter():
        n_a, n_b = edge
        print "phi_{0}{1} = ".format(n_a,n_b)
        print graph_name.edge[n_a][n_b]['edge_potential']

## utility to draw graphs

def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='lightgray', node_alpha=1.0,
               node_text_size=12,
               edge_color='lightgray', edge_alpha=1.0, edge_tickness=2,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(graph)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(graph)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(graph)
    else:
        graph_pos=nx.shell_layout(graph)

    nx.draw_networkx_nodes(graph,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(graph,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(graph, graph_pos,font_size=node_text_size,
                            font_family=text_font)
    sns.set_style("white")
    plt.axis('off')
    plt.show()
