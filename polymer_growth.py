from __future__ import print_function

import sys

import networkx as nx
import numpy as np
import itertools as it
import copy
import dill

import pickle
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import construct_example_mrfs as mrf
import exact_partition_functions as pf
import graph_utilities as gu
import belief_propogation as bp

np.seterr(all='warn')
np.seterr(under='ignore',over='ignore')

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import mdtraj as md

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


import functools
import openmm_potentials as omp


beta = omp.compute_beta(298)


# need to guard for infs/zeros in logs here
def safe_log(x):
    if x <= 0:
        return -np.inf
    return np.log(x)


# return the energy of a global configuration of the graph
def get_betaU_for_state(some_graph,global_state):

    all_node_pots = nx.get_node_attributes(some_graph,'node_potential')
    all_edge_pots = nx.get_edge_attributes(some_graph,'edge_potential')

    betaU_state_node = 0.0
    for node in some_graph.nodes_iter():
        phi_n = all_node_pots[node][global_state[node],0] # 0 cause col vector is 2-dim
        betaU_state_node += -safe_log(phi_n)

    betaU_state_edge = 0.0
    for edge in some_graph.edges_iter():
        n_a, n_b = edge
        phi_e = all_edge_pots[edge][global_state[n_a]][global_state[n_b]]
        betaU_state_edge += -safe_log(phi_e)

    betaU_state = (betaU_state_node+betaU_state_edge)

    return betaU_state

# shortcut for tuple sorting by energy
def get_best_states_ens(state_en_pairs,N=10):
    return sorted(state_en_pairs, key=lambda tup: tup[1])[:N]


# NONSTOCHASTIC
def polymer_growth_sampling(g,sample_size=10):
    """
    polymer growth sampling of a protein/peptide graph
    inputs:
        g: graph / markov random field of peptide/protein
        sample_size: number of states we downsample to as each residue is added
    outputs:
        best_state_en_pairs: list of state/energy tuples for the best sample_size states
    """
    # get the node indices
    node_indices = [n for n in g.nodes_iter()]
    for i in node_indices:

        # h is a temp graph with nodes/edges > i removed
        h = g.copy()
        h.remove_nodes_from(node_indices[i+1:])

        # if node zero, initialize polymer
        if i == 0:
            initial_states = [[state] for state in nx.get_node_attributes(g,'node_state_indices')[0]]
            state_en_pairs = [(state,get_betaU_for_state(h,state)) for state in initial_states]
            best_state_en_pairs = get_best_states_ens(state_en_pairs,N=sample_size)
            old_states = [state for (state,en) in best_state_en_pairs]

        # for other nodes, add the node to the existing polymer and resample
        else:
            new_node_states = nx.get_node_attributes(h,'node_state_indices')[i]
            new_state_en_pairs = []
            for old_state,new_node_state in it.product(old_states,new_node_states):
                new_tot_state = old_state+[new_node_state]
                new_state_en_pairs.append((new_tot_state, get_betaU_for_state(h,new_tot_state)))
                best_state_en_pairs = get_best_states_ens(new_state_en_pairs,N=sample_size)
                old_states = [state for (state,en) in best_state_en_pairs]

    return best_state_en_pairs


# blotzman sampling rather than just going for the lowest n energy states
def boltzman_sample_state_ens(state_betaE_list,sample_size=10):
    betaE_list = [energy for (state,energy) in state_betaE_list]
    state_list = [state for (state,energy) in state_betaE_list]
    state_inds = [i for (i,state_betaE) in enumerate(state_betaE_list)]

    beta_Emin = np.min(betaE_list)
    state_probs = np.exp( -(np.array(betaE_list)-beta_Emin) )
    state_probs /= np.sum(state_probs)

    my_sample = np.random.choice(state_inds, size=sample_size, replace=True, p=state_probs)
    state_betaE_sample = [state_betaE_list[i] for i in my_sample]

    return state_betaE_sample

# boltzmann sampling-based algorithm --  use this one
def stochastic_polymer_growth_sampling(g,sample_size=10):
    """
    polymer growth sampling of a protein/peptide graph
    inputs:
        g: graph / markov random field of peptide/protein
        sample_size: number of states we downsample to as each residue is added
    outputs:
        best_state_en_pairs: list of state/energy tuples for the sampled states
    """
    # get the node indices
    node_indices = [n for n in g.nodes_iter()]

    #successively add each node and downsample the product state-space
    for i in node_indices:

        # h is a temp graph with nodes/edges > i removed
        h = g.copy()
        h.remove_nodes_from(node_indices[i+1:])

        # if node zero, initialize polymer
        if i == 0:
            initial_states = [[state] for state in nx.get_node_attributes(g,'node_state_indices')[0]]
            state_en_pairs = [(state,get_betaU_for_state(h,state)) for state in initial_states]
            best_state_en_pairs = boltzman_sample_state_ens(state_en_pairs,sample_size=sample_size)
            old_states = [state for (state,en) in best_state_en_pairs]

        # for other nodes, add the node to the existing polymer and resample
        else:
            new_node_states = nx.get_node_attributes(h,'node_state_indices')[i]
            new_state_en_pairs = []

            # check all the new states we get by adding in a new node
            for old_state,new_node_state in it.product(old_states,new_node_states):
                new_tot_state = old_state+[new_node_state]
                new_state_en_pairs.append((new_tot_state, get_betaU_for_state(h,new_tot_state)))
                best_state_en_pairs = boltzman_sample_state_ens(new_state_en_pairs,sample_size=sample_size)
                old_states = [state for (state,en) in best_state_en_pairs]

    return best_state_en_pairs

# given a graph and a list of energies, calculate betaU
def get_betaU(graph,phys=False,sample_size=100):
    pg_sample = stochastic_polymer_growth_sampling(graph,sample_size=sample_size)
    betaU_list = [energy for (state,energy) in pg_sample]

    betaUmin = np.min(betaU_list)
    Z0 = np.sum([np.exp(-(betaU-betaUmin)) for betaU in betaU_list])

    betaUZ0 = np.sum([betaU*np.exp(-(betaU-betaUmin)) for betaU in betaU_list])
    betaU = betaUZ0/Z0
    if phys:
        pass

    return betaU

# given a graph and a list of energies, calculate betaTS
def get_betaTS(graph,phys=False,sample_size=100):
    pg_sample = stochastic_polymer_growth_sampling(graph,sample_size=sample_size)
    betaU_list = [energy for (state,energy) in pg_sample]

    betaUmin = np.min(betaU_list)
    bfacts = np.array([np.exp(-(betaU-betaUmin)) for betaU in betaU_list])
    p = bfacts/np.sum(bfacts)
    betaTS = -np.sum(p*np.log(p))

    if phys:
        d_c,d_h = graph.graph['num_chis']
        k_c,k_h = graph.graph['grid_points_per_chi']
        discretization_correction = d_c*np.log(k_c/(2*np.pi)) + d_h*np.log(k_h/(2*np.pi/3))
        betaTS -= discretization_correction

    return betaTS

# given a graph and a list of energies, calculate betaG
def get_betaG(graph,phys=False,sample_size=100):
    pg_sample = stochastic_polymer_growth_sampling(graph,sample_size=sample_size)
    betaU_list = [energy for (state,energy) in pg_sample]

    betaU = get_betaU(graph,phys=phys,sample_size=sample_size)
    betaTS = get_betaTS(graph,phys=phys,sample_size=sample_size)
    betaG = betaU - betaTS
    return betaG

# given a graph and a list of energies, calculate betaTS indirectly
def get_betaTS_indirect(graph,phys=False,sample_size=100):
    betaU = get_betaU(graph,phys=phys,sample_size=sample_size)
    betaG = get_betaG(graph,phys=phys,sample_size=sample_size)
    betaTS = betaU - betaG
    if phys:
        d_c,d_h = graph.graph['num_chis']
        k_c,k_h = graph.graph['grid_points_per_chi']
        discretization_correction = d_c*np.log(k_c/(2*np.pi)) + d_h*np.log(k_h/(2*np.pi/3))
        betaTS -= discretization_correction
    return betaTS
