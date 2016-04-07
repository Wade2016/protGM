## approximate partition functions

import networkx as nx
import numpy as np
import copy

# clip graph potentials to remove infs
def clean_up_graph(g,pot_min=0.0,pot_max = 1.0e+100):

    for edge in g.edges_iter():
        n,m = edge
        edge_pot = g.edge[n][m]['edge_potential']
        edge_pot = np.clip(edge_pot,pot_min,pot_max)
        g.edge[n][m]['edge_potential'] = edge_pot

    for node in g.nodes_iter():
        node_pot = g.node[node]['node_potential']
        node_pot = np.clip(node_pot,pot_min,pot_max)
        g.node[node]['node_potential'] = node_pot

def compute_delta(old_messages,new_messages, verbose=False):
    delta = 0.0
    for node_from in old_messages:
        for node_to in old_messages[node_from]:
            #print np.min(old_messages[node_from][node_to]), np.max(old_messages[node_from][node_to])
            delta += np.sum(np.fabs(new_messages[node_from][node_to] - old_messages[node_from][node_to]))
    if verbose:
        print 'delta =', delta
    return delta


# define a dict for old and new messages -- indexing is dict[from][to]
def initialize_messages(some_graph):
    old_messages = {}
    for node_from in some_graph.nodes_iter():
        old_messages[node_from] = {}
        for node_to in some_graph.neighbors(node_from):
            old_messages[node_from][node_to] = np.ones(some_graph.node[node_to]['node_potential'].shape)
    new_messages = copy.deepcopy(old_messages)
    return new_messages, old_messages


def calculate_node_beliefs(some_graph,messages):
    beliefs = {}
    for node in some_graph.nodes_iter():
        beliefs[node] = copy.deepcopy(some_graph.node[node]['node_potential'])
        for node_neighbor in some_graph.neighbors(node):
            beliefs[node] *= messages[node_neighbor][node]
        assert np.sum(beliefs[node]) >= 0.0
            # beliefs[node] = np.ones_like(beliefs[node])
        beliefs[node] = beliefs[node]/np.sum(beliefs[node])

    return beliefs


def calculate_edge_beliefs(some_graph,messages):
    beliefs = {}
    for edge in some_graph.edges_iter():
        i,j = edge
        if i>j:
            i,j = j,i

        x_i = copy.deepcopy(some_graph.node[i]['node_potential'])
        for node_neighbor in [n for n in some_graph.neighbors(i) if n != j]:
            x_i *= messages[node_neighbor][i]

        x_j = copy.deepcopy(some_graph.node[j]['node_potential'])
        for node_neighbor in [n for n in some_graph.neighbors(j) if n != i]:
            x_j *= messages[node_neighbor][j]

        beliefs[edge] = np.outer(x_i,x_j) * some_graph.edge[i][j]['edge_potential']

        assert np.sum(beliefs[edge]) >= 0.0
            # beliefs[edge] = np.ones_like(beliefs[edge])
        beliefs[edge] = beliefs[edge]/np.sum(beliefs[edge])

    return beliefs

###################
# Old BP indexing #
###################

# message passing for pairwise models, based on 14.31 in http://web.stanford.edu/~montanar/RESEARCH/BOOK/partD.pdf
def runBP_old(some_graph, N_BPiters=100, epsilon=1e-12, verbose=False):

    clean_up_graph(some_graph)

    deltas = []
    converged = False

    # initialize all messages as ones
    new_messages, old_messages = initialize_messages(some_graph)

    for N_BPiter in xrange(N_BPiters):
        for node_from in some_graph.nodes_iter():
            for node_to in some_graph.neighbors(node_from):
                nodes_gather = [n for n in some_graph.neighbors(node_from) if n != node_to]

                i,j = node_from,node_to  # indexing conventions from montanar paper

                # start out with node potential as message (ie modified uniform distribution);
                # if no neighbors other than target node, this is what gets passed
                x = copy.deepcopy(some_graph.node[i]['node_potential'])

                # if there are other neighbors, gather new messages from them
                if nodes_gather:
                    for k in nodes_gather:
                        x *= old_messages[k][i]

                # hit node i's info with the ij edge potential and pass to j
                psi_ij = copy.deepcopy(some_graph.edge[i][j]['edge_potential'])
                if j > i:
                    psi_ij = psi_ij.T

                new_messages[i][j] = psi_ij.dot(x)

                # normalize the new message
                assert np.sum(new_messages[i][j]) >= 0.0
                    # new_messages[i][j] = np.ones_like(new_messages[i][j])
                new_messages[i][j] = new_messages[i][j]/np.sum(new_messages[i][j])


        # check for convergence
        if verbose:
            print 'N_BPiter =', N_BPiter
        delta = compute_delta(old_messages,new_messages)
        deltas.append(delta)

        if delta < epsilon:
            converged = True
            break

        # get ready for the next iteration of BP, if needed
        old_messages = copy.deepcopy(new_messages)

    node_beliefs = calculate_node_beliefs(some_graph,new_messages)
    edge_beliefs = calculate_edge_beliefs(some_graph,new_messages)

    return node_beliefs,edge_beliefs,new_messages,converged,deltas

#######################
# End Old BP indexing #
#######################


######################################################
# re-write of bp function to agree with yfw indexing #
######################################################

# message passing for pairwise models, based on 14.31 in http://web.stanford.edu/~montanar/RESEARCH/BOOK/partD.pdf
def runBP(some_graph, N_BPiters=100, epsilon=1e-12, verbose=False):

    clean_up_graph(some_graph)

    deltas = []
    converged = False

    # initialize all messages as ones
    new_messages, old_messages = initialize_messages(some_graph)

    for N_BPiter in xrange(N_BPiters):
        if verbose:
            print N_BPiter
        for node_from in some_graph.nodes_iter():
            for node_to in some_graph.neighbors(node_from):
                nodes_gather = [n for n in some_graph.neighbors(node_from) if n != node_to]

                i,j = node_to,node_from
                #print "node_to = ", node_to, "node_from = ", node_from
                #print "to_pot = ", some_graph.node[node_to]['node_potential']
                #print "from_pot = ", some_graph.node[node_from]['node_potential']

                # start out with node potential as message (ie modified uniform distribution);
                # if no neighbors other than target node, this is what gets passed
                x = copy.deepcopy(some_graph.node[j]['node_potential'])

                # if there are other neighbors, gather new messages from them
                if nodes_gather:
                    for k in nodes_gather:
                        x *= old_messages[k][j] # messages are [from][to]

                # hit node i's info with the ij edge potential and pass to j
                psi_ij = copy.deepcopy(some_graph.edge[i][j]['edge_potential'])

                if j < i:
                    psi_ij = psi_ij.T

                #print "x = ", x
                #print "psi_ij = ", psi_ij

                new_messages[j][i] = psi_ij.dot(x) # node pots and messages are now col vectors
                if verbose:
                    if np.isnan(np.min(new_messages[j][i])):
                        # print psi_ij
                        print i,j
                        print x
                        print new_messages[j][i]

                # normalize the new message
                assert np.sum(new_messages[j][i]) >= 0.0, "message is f-ed up: %r" % new_messages[j][i]
                    # new_messages[j][i] = np.ones_like(new_messages[j][i])
                new_messages[j][i] = new_messages[j][i]/np.sum(new_messages[j][i])


        # check for convergence
        if verbose:
            print 'N_BPiter =', N_BPiter
        delta = compute_delta(old_messages,new_messages, verbose=verbose)
        deltas.append(delta)

        if delta < epsilon:
            converged = True
            break

        # get ready for the next iteration of BP, if needed
        old_messages = copy.deepcopy(new_messages)

    node_beliefs = calculate_node_beliefs(some_graph,new_messages)
    edge_beliefs = calculate_edge_beliefs(some_graph,new_messages)

    # print out min and max of each potential, trying to track down nan result
    #for k, v in node_beliefs.iteritems():
    #    print 'node = ', k, np.min(v),np.max(v)
    #for k, v in edge_beliefs.iteritems():
    #    print 'edge = ', k, np.min(v),np.max(v)

    return node_beliefs,edge_beliefs,new_messages,converged,deltas

##########################################################
# end re-write of bp function to agree with yfw indexing #
##########################################################


# temperature independent energy for nodes
def calculate_betaU_node(some_graph,node_beliefs):

    clean_up_graph(some_graph)

    betaU_node = 0
    for node in some_graph.nodes_iter():
        # only sum over nonzero entries
        nz_inds = node_beliefs[node] > 0
        betaU_this_node = np.sum( node_beliefs[node][nz_inds] * np.log(some_graph.node[node]['node_potential'][nz_inds]) )
        betaU_node -= betaU_this_node
        #print 'betaU_{0} = '.format(node), -betaU_this_node
    return betaU_node


# temperature independent energy for edges
def calculate_betaU_edge(some_graph,edge_beliefs):

    clean_up_graph(some_graph)

    betaU_edge = 0
    for edge in some_graph.edges_iter():
        i,j = edge
        nz_inds = edge_beliefs[edge] > 0
        betaU_this_edge = np.sum( edge_beliefs[edge][nz_inds] * np.log(some_graph.edge[i][j]['edge_potential'][nz_inds]) )
        betaU_edge -= betaU_this_edge
        #print 'betaU_{0} = '.format(edge), -betaU_this_edge
    return betaU_edge


# temperature independent energy for everything
def calculate_betaU_total(some_graph,node_beliefs,edge_beliefs):

    clean_up_graph(some_graph)

    betaU_node = calculate_betaU_node(some_graph,node_beliefs)
    betaU_edge = calculate_betaU_edge(some_graph,edge_beliefs)
    #if some_graph.graph['beta_min_energy']:
    #    print "beta_min_energy = ", some_graph.graph['beta_min_energy']
    betaU_total = betaU_node + betaU_edge# - some_graph.graph['beta_min_energy']
    return betaU_total


# temperature independent entropy for nodes
def calculate_betaTS_node(some_graph,node_beliefs):

    clean_up_graph(some_graph)

    betaTS_node = 0
    for node in some_graph.nodes_iter():
        q = len(some_graph.neighbors(node)) - 1
        nz_inds = node_beliefs[node] > 0
        betaTS_this_node = q*np.sum( node_beliefs[node][nz_inds]*np.log(node_beliefs[node][nz_inds]) )
        betaTS_node += betaTS_this_node
        #print 'betaTS_{0} = '.format(node), -betaTS_this_node
    return betaTS_node


# temperature independent entropy for edges
def calculate_betaTS_edge(some_graph,edge_beliefs):

    clean_up_graph(some_graph)

    betaTS_edge = 0
    for edge in some_graph.edges_iter():
        nz_inds = edge_beliefs[edge] > 0
        betaTS_this_edge = np.sum( edge_beliefs[edge][nz_inds] * np.log(edge_beliefs[edge][nz_inds]) )
        betaTS_edge -= betaTS_this_edge
        #print 'betaTS_{0} = '.format(edge), -betaTS_this_edge
    return betaTS_edge


# temperature independent entropy for everything
def calculate_betaTS_total(some_graph,node_beliefs,edge_beliefs,phys=False):

    clean_up_graph(some_graph)

    betaTS_node = calculate_betaTS_node(some_graph,node_beliefs)
    betaTS_edge = calculate_betaTS_edge(some_graph,edge_beliefs)
    betaTS_total = betaTS_node + betaTS_edge
    # only works for symmetry=3 right now
    if phys:
        d_c,d_h = some_graph.graph['num_chis']
        k_c,k_h = some_graph.graph['grid_points_per_chi']
        discretization_correction = d_c*np.log(k_c/(2*np.pi)) + d_h*np.log(k_h/(2*np.pi/3))
        # print('BP disc corr = ', discretization_correction)
        betaTS_total -= discretization_correction
     #print 'betaTS_total = ', betaTS_total
    return betaTS_total


# temperature independent free energy = -log(Z)
def calculate_betaG(some_graph, phys=False, verbose=False):

    clean_up_graph(some_graph)

    node_beliefs,edge_beliefs,messages,isconverged,deltapath = runBP(some_graph, verbose=verbose)
    betaU_total = calculate_betaU_total(some_graph,node_beliefs,edge_beliefs)
    betaTS_total = calculate_betaTS_total(some_graph,node_beliefs,edge_beliefs,phys=phys)
    betaG = betaU_total - betaTS_total
    return betaG


# temperature independent (as it should be) approximate partition function
def calculate_bethe_Z(some_graph, phys=False, verbose=False):

    clean_up_graph(some_graph)

    betaG = calculate_betaG(some_graph, phys=phys, verbose=verbose)
    Z_bethe = np.exp(-betaG)
    return Z_bethe


# explicit return of Energy and Entropy
def return_beta_UandTS(some_graph,phys=False):

    clean_up_graph(some_graph)

    node_beliefs,edge_beliefs,messages,isconverged,deltapath = runBP(some_graph)
    betaU_total = calculate_betaU_total(some_graph,node_beliefs,edge_beliefs)
    betaTS_total = calculate_betaTS_total(some_graph,node_beliefs,edge_beliefs,phys=phys)
    betaG = betaU_total - betaTS_total
    return {'betaU_total':betaU_total,'betaTS_total':betaTS_total}
