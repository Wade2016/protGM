from __future__ import print_function

import sys
from sys import stdout
import itertools as it

import dill
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import functools

import numpy as np

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import mdtraj as md

import networkx as nx


###############################
# Static dicts for chi angles #
###############################

# sets of chi angle atoms from mdtraj code
CHI1_ATOMS = [["N", "CA", "CB", "CG"],
              ["N", "CA", "CB", "CG1"],
              ["N", "CA", "CB", "SG"],
              ["N", "CA", "CB", "OG"],
              ["N", "CA", "CB", "OG1"]]
#CHI1_ATOMS_set = [set(atoms) for atoms in CHI1_ATOMS]

CHI2_ATOMS = [["CA", "CB", "CG", "CD"],
              ["CA", "CB", "CG", "CD1"],
              ["CA", "CB", "CG1", "CD1"],
              ["CA", "CB", "CG", "OD1"],
              ["CA", "CB", "CG", "ND1"],
              ["CA", "CB", "CG", "SD"]]
#CHI2_ATOMS_set = [set(atoms) for atoms in CHI2_ATOMS]

CHI3_ATOMS = [["CB", "CG", "CD", "NE"],
              ["CB", "CG", "CD", "CE"],
              ["CB", "CG", "CD", "OE1"],
              ["CB", "CG", "SD", "CE"]]
#CHI3_ATOMS_set = [set(atoms) for atoms in CHI3_ATOMS]

CHI4_ATOMS = [["CG", "CD", "NE", "CZ"],
              ["CG", "CD", "CE", "NZ"]]
#CHI4_ATOMS_set = [set(atoms) for atoms in CHI4_ATOMS]

ALL_CHIS = [CHI1_ATOMS,CHI2_ATOMS,CHI3_ATOMS,CHI4_ATOMS]


# dict of residue types and chi angle quartets
CHI_QUARTETS_RES_MAP = {
 'ACE': [],
 'ALA': [],
 'ARG': [['N', 'CA', 'CB', 'CG'],
  ['CA', 'CB', 'CG', 'CD'],
  ['CB', 'CG', 'CD', 'NE'],
  ['CG', 'CD', 'NE', 'CZ']],
 'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
 'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
 'CYS': [['N', 'CA', 'CB', 'SG']],
 'GLN': [['N', 'CA', 'CB', 'CG'],
  ['CA', 'CB', 'CG', 'CD'],
  ['CB', 'CG', 'CD', 'OE1']],
 'GLU': [['N', 'CA', 'CB', 'CG'],
  ['CA', 'CB', 'CG', 'CD'],
  ['CB', 'CG', 'CD', 'OE1']],
 'GLY': [],
 'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
 'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
 'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
 'LYS': [['N', 'CA', 'CB', 'CG'],
  ['CA', 'CB', 'CG', 'CD'],
  ['CB', 'CG', 'CD', 'CE'],
  ['CG', 'CD', 'CE', 'NZ']],
 'MET': [['N', 'CA', 'CB', 'CG'],
  ['CA', 'CB', 'CG', 'SD'],
  ['CB', 'CG', 'SD', 'CE']],
 'NME': [],
 'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
 'PRO': [],
 'SER': [['N', 'CA', 'CB', 'OG']],
 'THR': [['N', 'CA', 'CB', 'OG1']],
 'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
 'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
 'VAL': [['N', 'CA', 'CB', 'CG1']]
 }


# get all sets of downstream atoms for each of the chi angles
# need to deal with uncertain protonation states -- can't forget to move that hydrogen
# get chi_1 dowstream atome programatically (all in res not in backbone)
# then for the next chi, set differnce the first sublist (eg ['CB','HB2', 'HB3']) from chi_1 downstream
# continue subracting off for the next chi until the last one.
# note: won't actually use the last list for each res in the following dict

disjoint_chi_atom_names = {
    'ACE': [],
    'ALA': [],
    'ARG': [
        ['CB','HB2', 'HB3'],
        ['CG','HG2', 'HG3'],
        ['CD','HD2', 'HD3'],
        ['NE', 'CZ', 'NH1', 'NH2', 'HE', 'HH11', 'HH12', 'HH21', 'HH22']
        ],
    'ASN': [
        ['CB','HB2', 'HB3'],
        ['CG', 'ND2', 'OD1', 'HD21', 'HD22']
        ],
    'ASP': [
        ['CB','HB2', 'HB3'],
        ['CG', 'OD1', 'OD2']
        ],
    'CYS': [
        ['CB', 'SG', 'HB2', 'HB3', 'HG']
        ],
    'GLN': [
        ['CB', 'HB2', 'HB3'],
        ['CG','HG2', 'HG3'],
        ['CD', 'NE2', 'OE1', 'HE21', 'HE22']
        ],
    'GLU': [
        ['CB','HB2', 'HB3'],
        ['CG','HG2', 'HG3'],
        ['CD', 'OE1', 'OE2'] #prot state???
        ],
    'GLY': [],
    'HIS': [
        ['CB','HB2', 'HB3'],
        [ 'CG', 'CD2', 'ND1', 'CE1', 'NE2', 'HD1', 'HD2', 'HE1'] #prot state???
        ],
    'ILE': [
        ['CB','HB','CG2', 'HG21', 'HG22', 'HG23'],
        ['CG1',  'CD1',  'HG12', 'HG13', 'HD11', 'HD12', 'HD13']
        ],
    'LEU': [
        ['CB', 'HB2', 'HB3'],
        ['CG', 'CD1', 'CD2', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23']
        ],
    'LYS': [
        ['CB', 'HB2', 'HB3'],
        ['CG', 'HG2', 'HG3'],
        ['CD', 'HD2', 'HD3'],
        ['CE', 'NZ', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3']
        ],
    'MET': [
        ['CB', 'HB2', 'HB3'],
        ['CG', 'HG2', 'HG3'],
        ['SD', 'CE', 'HE1', 'HE2', 'HE3']
        ],
    'NME': [],
    'PHE': [
        ['CB', 'HB2', 'HB3'],
        ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'HD1', 'HD2', 'HE1', 'HE2', 'HZ']
        ],
    'PRO': [],
    'SER': [
        ['CB', 'OG', 'HB2', 'HB3', 'HG']
        ],
    'THR': [
        ['CB', 'CG2', 'OG1', 'HB', 'HG1', 'HG21', 'HG22', 'HG23']
        ],
    'TRP': [
        ['CB', 'HB2', 'HB3'],
        ['CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2',  'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2']
        ],
    'TYR': [
        ['CB', 'HB2', 'HB3'],
        ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'HD1', 'HD2', 'HE1', 'HE2', 'HH']
        ],
    'VAL': [
        ['CB', 'CG1', 'CG2', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23']
        ]
}

# hydrogen (methyl) chi angle degrees of fredom:
# TODO: figureout ambiguous nitrogen bs:
# ARG (double H), ASN (double H), GLN (double H), LYS (triple H)

H_CHI_QUARTETS_RES_MAP = {
 'ACE': [['O','C','CH3','H1']],
 'ALA': [['N','CA','CB','HB1']],
 'ARG': [],
 'ASN': [],
 'ASP': [],
 'CYS': [],
 'GLN': [],
 'GLU': [],
 'GLY': [],
 'HIS': [],
 'ILE': [['CA','CB','CG2','HG21'],['CB','CG1','CD1','HD11']],
 'LEU': [['CB','CG','CD1','HD11'],['CB','CG','CD2','HD21']],
 'LYS': [],
 'MET': [['CG','SD','CE','HE1']],
 'NME': [['H','N','C','H1']],
 'PHE': [],
 'PRO': [],
 'SER': [],
 'THR': [['CA','CB','CG2','HG21']],
 'TRP': [],
 'TYR': [],
 'VAL': [['CA','CB','CG1','HG11'],['CA','CB','CG2','HG21']]
 }

H_CHI_DOWNSTREAM_ATM_NAMES = {
 'ACE': [['CH3','H1','H2','H3']],
 'ALA': [['CB','HB1','HB2','HB3']],
 'ARG': [],
 'ASN': [],
 'ASP': [],
 'CYS': [],
 'GLN': [],
 'GLU': [],
 'GLY': [],
 'HIS': [],
 'ILE': [['CG2','HG21','HG22','HG23'],['CD1','HD11','HD12','HD13']],
 'LEU': [['CD1','HD11','HD12','HD13'],['CD2','HD21','HD22','HD23']],
 'LYS': [],
 'MET': [['CE','HE1','HE2','HE3']],
 'NME': [['C','H1','H2','H3']],
 'PHE': [],
 'PRO': [],
 'SER': [],
 'THR': [['CG2','HG21','HG22','HG23']],
 'TRP': [],
 'TYR': [],
 'VAL': [['CG1','HG11','HG12','HG13'],['CG2','HG21','HG22','HG23']]
}

#############################
# basic geometric functions #
#############################

def rotation_matrix(angle, direction):
    """
    Return 4x4 matrix to rotate about axis defined by an angle and a direction.
    Adapted from transformations.py package.
    """

    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction /= np.linalg.norm(direction)

    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                   [ direction[2], 0.0,          -direction[0]],
                   [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    return M


def rot_around_vec_at_point(data,theta,vec,point):
    """
    data: nx3 list of triples of the points that are getting rotated
    theta: angle in radians by which to rotate data
    vec: 1x3 vector around which to rotate
    point: 1x3 point at which to apply the rotation (eg not the origin)

    returns: nx3 numpy array of points that were rotated
    """

    # use rotation_matrix from transformations.py to check my version
    R = rotation_matrix(theta,vec)[:3,:3]
    data_centered = data - point
    data_centered_rotated = R.dot(data_centered.T).T
    data_centered_rotated_movedback = data_centered_rotated + point

    return data_centered_rotated_movedback


# from mathoverflow: Rahul (http://math.stackexchange.com/users/856/rahul)
# How do I calculate a dihedral angle given Cartesian coordinates?
# URL (version: 2011-06-23): http://math.stackexchange.com/q/47084
def compute_dihedral(W,X,Y,Z):
    """
    Computes a dihedral angle from four input points.
    The impliedtopolog is W-X-Y-Z, ie X-Y is the central axis of the angle.
    W,X,Y,Z should all be length 3 numpy arrays of floats.
    """

    b1 = X - W
    b2 = Y - X
    b3 = Z - Y

    b1_n = b1/np.linalg.norm(b1)
    b2_n = b2/np.linalg.norm(b2)
    b3_n = b3/np.linalg.norm(b3)

    n1 = np.cross(b1_n,b2_n)
    n2 = np.cross(b2_n,b3_n)
    m1 = np.cross(n1,b2_n)

    x = np.dot(n1,n2)
    y = np.dot(m1,n2)

    d = np.arctan2(y,x)

    return d



#####################
# utility functions #
#####################

# utility functions to make working wth pdb object a little easier
# default to working with indices rather thatn ids; id = index + 1

# uniform chi grid -- jut an evenly spaced list on [0,2pi)
def chi_grid(Npoints):
    return [(2.0*np.pi)*i/Npoints for i in xrange(Npoints)]

# uniform chi grid for methyl chis-- jut an evenly spaced list on [0,2pi/3)
def h_chi_grid(Npoints,symmetry=3):
    return [(2.0*np.pi/symmetry)*i/Npoints for i in xrange(Npoints)]

# returns a list of cartesian product of grid points for all the chi and h_chi angles
# use as: for i,(chi_angle_tuple,h_chi_angle_tuple) in enumerate(total_chi_grid):
# if residue is not selected for use, rutrn an empty grid so it stays put
def get_total_chi_grid(N_chi_samples, N_h_chi_samples, res_index, simulation, h_chi_symmetry=3, res_select=None):
    if res_select is not None: #if there's a list of selected residues
        if res_index in res_select: # sample densely if res is in list
            res_type = get_res_name(res_index,simulation)
            chi_grid_list = [chi_grid(N_chi_samples) for chi in CHI_QUARTETS_RES_MAP[res_type]]
            h_chi_grid_list = [h_chi_grid(N_h_chi_samples,symmetry=h_chi_symmetry) for h_chi in H_CHI_QUARTETS_RES_MAP[res_type]]
        else: # or don't give it a grid at all if it's nots
            chi_grid_list = []
            h_chi_grid_list = []
    else: # if res_select is None, sample every residue equally
        res_type = get_res_name(res_index,simulation)
        chi_grid_list = [chi_grid(N_chi_samples) for chi in CHI_QUARTETS_RES_MAP[res_type]]
        h_chi_grid_list = [h_chi_grid(N_h_chi_samples,symmetry=h_chi_symmetry) for h_chi in H_CHI_QUARTETS_RES_MAP[res_type]]

    return list(it.product(it.product(*chi_grid_list), it.product(*h_chi_grid_list)))

# atom indices for a specific residue
def get_atom_inds(res_index,pdb):
    # can't figure out how to save off residue object not as an iterator
    for res in pdb.topology.residues():
        if res.index == res_index:
            return [a.index for a in res.atoms()]

# res_indices is a list: [i,j,k,etc]. if only want res i, need to pass as [i]
# replace get_atom_inds eventually?
def get_atom_inds_gen(res_indices,pdb):
    # can't figure out how to save off residue object not as an iterator
    atom_list = []
    for res in pdb.topology.residues():
        if res.index in res_indices:
            atom_list.extend([a.index for a in res.atoms()])
    return atom_list

# atom names for a specific residue
def get_atom_names(res_index,pdb):
    # can't figure out how to save off residue object not as an iterator
    for res in pdb.topology.residues():
        if res.index == res_index:
            return [a.name for a in res.atoms()]

# zipped list of indices and names in residue
def get_atom_inds_names(res_index,pdb):
    atom_names = get_atom_names(res_index,pdb)
    atom_inds = get_atom_inds(res_index,pdb)
    return zip(atom_inds,atom_names)

# name of residue number
def get_res_name(res_index,pdb):
    # can't figure out how to save off residue object not as an iterator
    for res in pdb.topology.residues():
        if res.index == res_index:
            return res.name

# all atom names in molecule, indexed
def get_all_atom_names(pdb):
    all_atom_names = [0]*len(pdb.getPositions())
    for a in pdb.topology.atoms():
        all_atom_names[a.index] = a.name
    return all_atom_names


# all atom names in molecule, indexed
def get_all_atom_inds(pdb):
    all_atom_inds = [i for (i,a) in enumerate(pdb.topology.atoms())]
    return all_atom_inds

# backbone atoms: ['N', 'H', 'CA', 'HA', 'C', 'O']
def get_bb_atoms(res_index,pdb):
    res_type = get_res_name(res_index,pdb)

    if res_type is not 'PRO':
        return ['N', 'H', 'CA', 'HA', 'C', 'O']
    else: # should this be modified for proline ???
        return ['N', 'H', 'CA', 'HA', 'C', 'O']


#######################
# chi angle functions #
#######################

# atoms 'downstream' of chi that get rotated for different residues
# if A-B-C-D are the atoms determining the diherdral angle, only C, D and
# atoms downstream of them are roTated.  Though both B and C don't move, atoms
# (eg hydrogens) attached to C will be rotated.

def get_chi_atom_inds_names(res_index,pdb,chi=1):
    all_atom_inds_names_in_res = get_atom_inds_names(res_index,pdb)
    all_atom_inds_in_res = [i for (i,a) in all_atom_inds_names_in_res]
    all_atoms_names_in_res = [a for (i,a) in all_atom_inds_names_in_res]

    res_type = get_res_name(res_index,pdb)
    if res_type in ['ALA','GLY','PRO','ACE','NME']:
        chi_atom_inds_names = None
    else: # scan through list of all possible chi angle heavy atoms for chi-N
        for atom_list in ALL_CHIS[chi-1]:
            if set(atom_list).issubset(all_atoms_names_in_res):
                chi_atom_names = [a_name for (a_ind,a_name) in all_atom_inds_names_in_res if a_name in atom_list]
                chi_atom_inds = [a_ind for (a_ind,a_name) in all_atom_inds_names_in_res if a_name in atom_list]
        chi_atom_inds_names = zip(chi_atom_inds,chi_atom_names)

    return chi_atom_inds_names


# Get all chi atom quartets all at once
def get_chi_quartet_inds_names(res_index,pdb):
    res_type = get_res_name(res_index,pdb)
    all_chi_quartet_names = CHI_QUARTETS_RES_MAP[res_type]

    all_atom_inds_names_in_res = get_atom_inds_names(res_index,pdb)
    all_atom_inds_in_res = [i for (i,a) in all_atom_inds_names_in_res]
    all_atoms_names_in_res = [a for (i,a) in all_atom_inds_names_in_res]

    all_chi_quartet_inds_names = []
    for chi_quartet_names in all_chi_quartet_names:
        if set(chi_quartet_names).issubset(all_atoms_names_in_res):
            chi_atom_names = [a_name for (a_ind,a_name) in all_atom_inds_names_in_res if a_name in chi_quartet_names]
            chi_atom_inds = [a_ind for (a_ind,a_name) in all_atom_inds_names_in_res if a_name in chi_quartet_names]
            chi_atom_inds_names = zip(chi_atom_inds,chi_atom_names)
            chi_atom_inds_names.sort(key=lambda (ind,name): chi_quartet_names.index(name)) # keep geometry correct
            all_chi_quartet_inds_names.append(chi_atom_inds_names)

    return all_chi_quartet_inds_names


# Get all h-chi atom quartets all at once
def get_h_chi_quartet_inds_names(res_index,pdb):
    res_type = get_res_name(res_index,pdb)
    all_h_chi_quartet_names = H_CHI_QUARTETS_RES_MAP[res_type]
    all_atom_inds_names_in_res = get_atom_inds_names(res_index,pdb)

    all_atom_inds_in_res = [i for (i,a) in all_atom_inds_names_in_res]
    all_atoms_names_in_res = [a for (i,a) in all_atom_inds_names_in_res]

    all_chi_quartet_inds_names = []
    for chi_quartet_names in all_h_chi_quartet_names:
        chi_atom_names = [a_name for (a_ind,a_name) in all_atom_inds_names_in_res if a_name in chi_quartet_names]
        chi_atom_inds = [a_ind for (a_ind,a_name) in all_atom_inds_names_in_res if a_name in chi_quartet_names]
        chi_atom_inds_names = zip(chi_atom_inds,chi_atom_names)
        chi_atom_inds_names.sort(key=lambda (ind,name): chi_quartet_names.index(name)) # keep geometry correct
        all_chi_quartet_inds_names.append(chi_atom_inds_names)

    return all_chi_quartet_inds_names

# res_select is either None or a list of res_indeces to use
# if None, use all residues, else use only those in list
def get_num_chis_and_hchis(res_index,pdb,res_select=None):
    if res_select is None:
        n_chis = len(get_chi_quartet_inds_names(res_index,pdb))
        n_hchis = len(get_h_chi_quartet_inds_names(res_index,pdb))
    else:
        if res_index in res_select:
            n_chis = len(get_chi_quartet_inds_names(res_index,pdb))
            n_hchis = len(get_h_chi_quartet_inds_names(res_index,pdb))
        else:
            n_chis = 0
            n_hchis = 0
    return (n_chis,n_hchis)

def get_all_h_chi_downstream_atom_inds_names(res_index,pdb):
    res_type = get_res_name(res_index,pdb)
    all_atom_inds_names_in_res = get_atom_inds_names(res_index,pdb)
    full_ds_ats = H_CHI_DOWNSTREAM_ATM_NAMES[res_type]
    full_ds_ats_with_inds = [[(i,a) for (i,a) in all_atom_inds_names_in_res if a in l] for l in full_ds_ats]
    return full_ds_ats_with_inds


# chi_1 is easy, the downstream atoms are just the atoms not in the backbone
def get_chi_1_downstream_atom_names(res_index,pdb):
    res_type = get_res_name(res_index,pdb)
    if res_type in ['ALA','GLY','PRO','ACE','NME']:
        chi_1_downstream_atom_names = None
    else:
        chi_1_downstream_atom_names = [a for a in get_atom_names(res_index,pdb) if a not in get_bb_atoms(res_index,pdb)]
    return chi_1_downstream_atom_names


def get_chi_1_downstream_atom_inds(res_index,pdb):
    res_type = get_res_name(res_index,pdb)
    if  res_type in ['ALA','GLY','PRO','ACE','NME']:
        chi_1_downstream_atom_inds = None
    else:
        chi_1_downstream_atom_names = get_chi_1_downstream_atom_names(res_index,pdb)
        all_atom_inds_names = get_atom_inds_names(res_index,pdb)
        chi_1_downstream_atom_inds = [i for (i,a) in all_atom_inds_names if a in chi_1_downstream_atom_names]
    return chi_1_downstream_atom_inds


# have to do this wonky set subtraction beace we don't know the names of all the atoms
# at the tips of the residues, since protonation is weird.   so get all atoms downstream
# of chi_1 by subracting off the backbone atoms, and work form there.  besides the terminal
# hydrogens, the other atoms have predictible names.

def get_all_chi_downstream_atom_inds_names(res_index,pdb):
    res_type = get_res_name(res_index,pdb)
    disj_ds_ats = disjoint_chi_atom_names[res_type]
    chi_1_ds_at_names = get_chi_1_downstream_atom_names(res_index,pdb)

    all_atom_inds_names_in_res = get_atom_inds_names(res_index,pdb)
    all_atom_inds_in_res = [i for (i,a) in all_atom_inds_names_in_res]
    all_atoms_names_in_res = [a for (i,a) in all_atom_inds_names_in_res]

    # needed if get_chi_1_downstream_atom_names returns None instead of empty list
    if chi_1_ds_at_names is None:
        chi_1_ds_at_names = []

    full_ds_ats = []
    temp_set = set(chi_1_ds_at_names)

    # start with the full downstream list, for ch_1, and iterative subtract off
    # atoms that aren't downstream of the latter chi angles
    for chi_i in xrange(len(disj_ds_ats)):
        full_ds_ats.append(list(temp_set))
        temp_set = temp_set.difference(set(disj_ds_ats[chi_i]))

    full_ds_ats = [sorted(l) for l in full_ds_ats]
    full_ds_ats_with_inds = [[(i,a) for (i,a) in all_atom_inds_names_in_res if a in l] for l in full_ds_ats]

    return full_ds_ats_with_inds


# get the chi angle specified by chi=N in the simulation
def get_chi(res_index,simulation,chi=1,all_positions=None):
    chi_atom_inds_names = get_chi_atom_inds_names(res_index,simulation,chi=1)
    chi_atom_names = [a_name for (a_ind,a_name) in chi_atom_inds_names]
    chi_atom_inds = [a_ind for (a_ind,a_name) in chi_atom_inds_names]

    if all_positions is not None:
        all_atom_positions = all_positions
    else:
        all_atom_positions = simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometers)

    chi_atom_positions = [all_atom_positions[i] for i in chi_atom_inds]
    dihedral_angle = compute_dihedral(*chi_atom_positions)
    while dihedral_angle < 0.0:
        dihedral_angle += 2.0*np.pi
    while dihedral_angle >= 2.0*np.pi:
        dihedral_angle -= 2.0*np.pi
    return dihedral_angle

# get all chis for a residue at once
def get_all_chis(res_index,simulation,all_positions=None):
    all_chi_atom_inds_names = get_chi_quartet_inds_names(res_index,simulation)
    all_chi_atom_names = [[a_name for (a_ind,a_name) in chi_q] for chi_q in all_chi_atom_inds_names]
    all_chi_atom_inds = [[a_ind for (a_ind,a_name) in chi_q] for chi_q in all_chi_atom_inds_names]

    if all_positions is not None:
        all_atom_positions = all_positions
    else:
        all_atom_positions = simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometers)

    dihedral_angles = []
    for chi_i,chi_atom_inds in enumerate(all_chi_atom_inds):
        chi_atom_positions = [np.array(all_atom_positions[i]) for i in chi_atom_inds]
        # print(chi_atom_positions)
        dihedral_angle = compute_dihedral(*chi_atom_positions)

        while dihedral_angle < 0.0:
            dihedral_angle += 2.0*np.pi
        while dihedral_angle >= 2.0*np.pi:
            dihedral_angle -= 2.0*np.pi

        dihedral_angles.append(dihedral_angle)

    return dihedral_angles


# get all h-chis for a residue at once
def get_all_h_chis(res_index,simulation,all_positions=None):
    all_h_chi_atom_inds_names = get_h_chi_quartet_inds_names(res_index,simulation)
    # print(all_h_chi_atom_inds_names)
    all_h_chi_atom_names = [[a_name for (a_ind,a_name) in chi_q] for chi_q in all_h_chi_atom_inds_names]
    all_h_chi_atom_inds = [[a_ind for (a_ind,a_name) in chi_q] for chi_q in all_h_chi_atom_inds_names]

    if all_positions is not None:
        all_atom_positions = all_positions
    else:
        all_atom_positions = simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometers)

    dihedral_angles = []
    for h_chi_i,h_chi_atom_inds in enumerate(all_h_chi_atom_inds):
        h_chi_atom_positions = [np.array(all_atom_positions[i]) for i in h_chi_atom_inds]
        # print(h_chi_atom_positions)
        dihedral_angle = compute_dihedral(*h_chi_atom_positions)

        while dihedral_angle < 0.0:
            dihedral_angle += 2.0*np.pi
        while dihedral_angle >= 2.0*np.pi:
            dihedral_angle -= 2.0*np.pi

        dihedral_angles.append(dihedral_angle)

    return dihedral_angles

# add random offset to angle?
def set_chi_1(angle,res_index,simulation,debug=False,all_positions=None):
    res_type = get_res_name(res_index,simulation)
    if res_type not in ['ALA','GLY','PRO','ACE','NME']:

        # not yet, but maybe passing postions could save tome?
        if all_positions is not None:
            p = all_positions
        else:
            p = simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometers)

        initial_chi = get_chi(res_index,simulation,chi=1,all_positions=p);
        if debug: print(initial_chi)
        rot_angle = initial_chi - angle;
        if debug: print(rot_angle)
        res_type = get_res_name(res_index,simulation);
        if debug: print(res_type)

        all_atom_inds_names_in_res = get_atom_inds_names(res_index,simulation);
        if debug: print(all_atom_inds_names_in_res)
        all_atom_inds_in_res = [i for (i,a) in all_atom_inds_names_in_res];
        if debug: print(all_atom_inds_in_res)
        all_atoms_names_in_res = [a for (i,a) in all_atom_inds_names_in_res];
        if debug: print(all_atoms_names_in_res)

        chi_1_downstream_atom_inds = get_chi_1_downstream_atom_inds(res_index,simulation);
        if debug: print(chi_1_downstream_atom_inds)
        chi_1_downstream_atom_pos = np.array([p[i] for i in chi_1_downstream_atom_inds]);
        if debug: print(chi_1_downstream_atom_pos)

        chi_atom_inds_names = get_chi_atom_inds_names(res_index,simulation,chi=1);
        if debug: print(chi_atom_inds_names)
        chi_atom_names = [a_name for (a_ind,a_name) in chi_atom_inds_names];
        if debug: print(chi_atom_names)
        chi_atom_inds = [a_ind for (a_ind,a_name) in chi_atom_inds_names];
        if debug: print(chi_atom_inds)
        chi_atom_positions = np.array([p[i] for i in chi_atom_inds]);
        if debug: print(chi_atom_positions)

        A,B,C,D = np.array(chi_atom_positions)
        v = C-B
        rotated_points = rot_around_vec_at_point(chi_1_downstream_atom_pos,rot_angle,v,C)
        rotated_inds_points = zip(chi_1_downstream_atom_inds,[tuple(l) for l in rotated_points])

        for (i,pos) in rotated_inds_points:
            if debug: print(i,p[i])
            p[i] = pos
            if debug: print(p[i])

        q = Quantity(p, unit=nanometer)

        simulation.context.setPositions(q)


# set all heavy atom chis as well as methyl group chis all at once
def set_all_chis_and_h_chis(angle_tuple,h_angle_tuple,res_index,simulation,all_positions=None,res_select=None,debug=False):

    # not yet, but maybe passing postions could save time?
    if all_positions is not None:
        p = all_positions
    else:
        p = simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometers)

    # set_all_chis(angle_tuple,res_index,simulation,debug=False,all_positions=p)
    # set_all_h_chis(h_angle_tuple,res_index,simulation,debug=False,all_positions=p)

    # only set chi if residue is selected or if there is no selection list
    if res_select is None:
        set_all_chis(angle_tuple,res_index,simulation,debug=False,all_positions=p)
        set_all_h_chis(h_angle_tuple,res_index,simulation,debug=False,all_positions=p)
    else:
        if res_index in res_select:
            set_all_chis(angle_tuple,res_index,simulation,debug=False,all_positions=p)
            set_all_h_chis(h_angle_tuple,res_index,simulation,debug=False,all_positions=p)
        else:
            pass


# set all chis at once
def set_all_chis(angle_tuple,res_index,simulation,debug=False,all_positions=None):

    res_type = get_res_name(res_index,simulation)

    if res_type not in ['ALA','GLY','PRO','ACE','NME']:

        all_atom_inds_names_in_res = get_atom_inds_names(res_index,simulation)
        all_atom_inds_in_res = [i for (i,a) in all_atom_inds_names_in_res]
        all_atoms_names_in_res = [a for (i,a) in all_atom_inds_names_in_res]

        # not yet, but maybe passing postions could save time?
        if all_positions is not None:
            p = all_positions
        else:
            p = simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometers)

        all_chi_quartet_inds_names = get_chi_quartet_inds_names(res_index,simulation)
        initial_chis = get_all_chis(res_index,simulation,all_positions=p)

        all_chi_downstream_atom_inds_names = get_all_chi_downstream_atom_inds_names(res_index,simulation)
        all_chi_downstream_atom_inds = [[a_ind for (a_ind,a_name) in ds_ats] for ds_ats in all_chi_downstream_atom_inds_names]
        all_chi_downstream_atom_names = [[a_name for (a_ind,a_name) in ds_ats] for ds_ats in all_chi_downstream_atom_inds_names]


        for chi_i,initial_chi in enumerate(initial_chis):
            angle = angle_tuple[chi_i]
            rot_angle = initial_chi - angle

            chi_i_downstream_atom_inds = all_chi_downstream_atom_inds[chi_i]
            chi_i_downstream_atom_pos = np.array([p[i] for i in chi_i_downstream_atom_inds])

            chi_i_atom_inds_names = all_chi_quartet_inds_names[chi_i]
            chi_i_atom_names = [a_name for (a_ind,a_name) in chi_i_atom_inds_names]
            chi_i_atom_inds = [a_ind for (a_ind,a_name) in chi_i_atom_inds_names]
            chi_i_atom_positions = np.array([p[i] for i in chi_i_atom_inds])

            A,B,C,D = np.array(chi_i_atom_positions)
            v = C-B
            rotated_points = rot_around_vec_at_point(chi_i_downstream_atom_pos,rot_angle,v,C)
            rotated_inds_points = zip(chi_i_downstream_atom_inds,[tuple(l) for l in rotated_points])

            for (i,pos) in rotated_inds_points:
                if debug: print(i,p[i])
                p[i] = pos
                if debug: print(p[i])

        q = Quantity(p, unit=nanometer)
        simulation.context.setPositions(q)


def set_all_h_chis(h_angle_tuple,res_index,simulation,debug=False,all_positions=None):

    res_type = get_res_name(res_index,simulation)

    if len(h_angle_tuple) > 0:

        all_atom_inds_names_in_res = get_atom_inds_names(res_index,simulation)
        all_atom_inds_in_res = [i for (i,a) in all_atom_inds_names_in_res]
        all_atoms_names_in_res = [a for (i,a) in all_atom_inds_names_in_res]

        # not yet, but maybe passing postions could save time?
        if all_positions is not None:
            p = all_positions
        else:
            p = simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometers)

        all_h_chi_quartet_inds_names = get_h_chi_quartet_inds_names(res_index,simulation)
        initial_h_chis = get_all_h_chis(res_index,simulation,all_positions=p)

        all_h_chi_downstream_atom_inds_names = get_all_h_chi_downstream_atom_inds_names(res_index,simulation)
        all_h_chi_downstream_atom_inds = [[a_ind for (a_ind,a_name) in ds_ats] for ds_ats in all_h_chi_downstream_atom_inds_names]
        all_h_chi_downstream_atom_names = [[a_name for (a_ind,a_name) in ds_ats] for ds_ats in all_h_chi_downstream_atom_inds_names]

        for h_chi_i,initial_h_chi in enumerate(initial_h_chis):
            h_angle = h_angle_tuple[h_chi_i]
            h_rot_angle = initial_h_chi - h_angle

            h_chi_i_downstream_atom_inds = all_h_chi_downstream_atom_inds[h_chi_i]
            h_chi_i_downstream_atom_pos = np.array([p[i] for i in h_chi_i_downstream_atom_inds])

            h_chi_i_atom_inds_names = all_h_chi_quartet_inds_names[h_chi_i]
            h_chi_i_atom_names = [a_name for (a_ind,a_name) in h_chi_i_atom_inds_names]
            h_chi_i_atom_inds = [a_ind for (a_ind,a_name) in h_chi_i_atom_inds_names]
            h_chi_i_atom_positions = np.array([p[i] for i in h_chi_i_atom_inds])

            h_A,h_B,h_C,h_D = np.array(h_chi_i_atom_positions)
            h_v = h_C-h_B
            h_rotated_points = rot_around_vec_at_point(h_chi_i_downstream_atom_pos,h_rot_angle,h_v,h_C)
            h_rotated_inds_points = zip(h_chi_i_downstream_atom_inds,[tuple(h_l) for h_l in h_rotated_points])

            for (h_i,h_pos) in h_rotated_inds_points:
                if debug: print(h_i,p[h_i])
                p[h_i] = h_pos
                if debug: print(p[h_i])

            q = Quantity(p, unit=nanometer)
            simulation.context.setPositions(q)


#################################
# energy mainpulation functions #
#################################

def get_all_force_parameters(system):

    params = {'bond_force_params' : {},
              'angle_force_params' : {},
              'torsion_force_params' : {},
              'nonbonded_force_params' : {},
              'nonbonded_exceptions_params' : {}}

    for force in system.getForces():

        if isinstance(force, HarmonicBondForce):
            for bondIndex in xrange(force.getNumBonds()):
                params['bond_force_params'][bondIndex] = force.getBondParameters(bondIndex)

        elif isinstance(force, HarmonicAngleForce):
            for angleIndex in xrange(force.getNumAngles()):
                params['angle_force_params'][angleIndex] = force.getAngleParameters(angleIndex)

        elif isinstance(force, PeriodicTorsionForce):
            for torsionIndex in xrange(force.getNumTorsions()):
                params['torsion_force_params'][torsionIndex] = force.getTorsionParameters(torsionIndex)

        elif isinstance(force, NonbondedForce):
            for particleIndex in xrange(force.getNumParticles()):
                params['nonbonded_force_params'][particleIndex] = force.getParticleParameters(particleIndex)

            for exceptionIndex in xrange(force.getNumExceptions()):
                params['nonbonded_exceptions_params'][exceptionIndex] = force.getExceptionParameters(exceptionIndex)

        else:
            raise ValueError("{0} is not a force that is handled in this function yet.".format(force))

    return params


def set_all_force_parameters(system,simulation,params):

    for force in system.getForces():

        if isinstance(force, HarmonicBondForce):
            force_param_type = 'bond_force_params'
            for bondIndex in xrange(force.getNumBonds()):
                force.setBondParameters(bondIndex,*params[force_param_type][bondIndex])

        elif isinstance(force, HarmonicAngleForce):
            force_param_type = 'angle_force_params'
            for angleIndex in xrange(force.getNumAngles()):
                force.setAngleParameters(angleIndex,*params[force_param_type][angleIndex])

        elif isinstance(force, PeriodicTorsionForce):
            force_param_type = 'torsion_force_params'
            for torsionIndex in xrange(force.getNumTorsions()):
                force.setTorsionParameters(torsionIndex,*params[force_param_type][torsionIndex])

        elif isinstance(force, NonbondedForce):
            force_param_type = 'nonbonded_force_params'
            for particleIndex in xrange(force.getNumParticles()):
                force.setParticleParameters(particleIndex,*params[force_param_type][particleIndex])

            for exceptionIndex in xrange(force.getNumExceptions()):
                force_param_type = 'nonbonded_exceptions_params'
                force.setExceptionParameters(exceptionIndex,*params[force_param_type][exceptionIndex])

        else:
            raise ValueError("{0} is not a force that is handled in this function yet.".format(force))

        force.updateParametersInContext(simulation.context)


def zero_all_forces(system,simulation):

    for force in simulation.system.getForces():

        if isinstance(force, HarmonicBondForce):
            for bondIndex in xrange(force.getNumBonds()):
                a1,a2,bond_length,bond_k = force.getBondParameters(bondIndex)
                force.setBondParameters(bondIndex,a1,a2,1,0)

        elif isinstance(force, HarmonicAngleForce):
            for angleIndex in xrange(force.getNumAngles()):
                a1,a2,a3,equilib_angle,angle_k = force.getAngleParameters(angleIndex)
                force.setAngleParameters(angleIndex,a1,a2,a3,equilib_angle,0)

        elif isinstance(force, PeriodicTorsionForce):
            for torsionIndex in xrange(force.getNumTorsions()):
                a1,a2,a3,a4,tors_period,tors_phase,tors_k = force.getTorsionParameters(torsionIndex)
                force.setTorsionParameters(torsionIndex,a1,a2,a3,a4,1,0,0)

        elif isinstance(force, NonbondedForce):
            for particleIndex in xrange(force.getNumParticles()):
                force.setParticleParameters(particleIndex,0,1,0)

            for exceptionIndex in xrange(force.getNumExceptions()):
                (a1,a2,chargeProd,sigma,epsilon) = force.getExceptionParameters(exceptionIndex)
                force.setExceptionParameters(exceptionIndex,a1,a2,100*sys.float_info.epsilon*chargeProd,1,100*sys.float_info.epsilon*epsilon)

            # for exceptionIndex in xrange(force.getNumExceptions()):
            #     (a1,a2,chargeProd,sigma,epsilon) = force.getExceptionParameters(exceptionIndex)
            #     force.setExceptionParameters(exceptionIndex,a1,a2,0,1,0)

        else:
            raise ValueError("{0} is not a force that is handled in this function yet.".format(force))

        force.updateParametersInContext(simulation.context)


# given a vacuum simulation, fake a solvent with a uniform dielectric
def turn_on_dielectric(system,simulation,eps_rel=60.0):

    for force in simulation.system.getForces():

        if isinstance(force, NonbondedForce):
            for particleIndex in xrange(force.getNumParticles()):
                charge, sigma, epsilon = force.getParticleParameters(particleIndex)
                force.setParticleParameters(particleIndex,charge/np.sqrt(eps_rel),sigma,epsilon)

            for exceptionIndex in xrange(force.getNumExceptions()):
                (a1,a2,chargeProd,sigma,epsilon) = force.getExceptionParameters(exceptionIndex)
                force.setExceptionParameters(exceptionIndex,a1,a2,chargeProd/eps_rel,sigma,epsilon)

        force.updateParametersInContext(simulation.context)


# TODO: make a dict of atom indices and store the bonds, angles, dihedrals, etc
# that they're in, so only have to iterate over those bonds angles etc ?

# shared / dangling bonded terms -- exclude all and put in edge energy?

# node potentials need bonded and nonbonded forces -- edge potentials only need nonbonded
def turn_on_node(res_index,simulation,zeroed_system,orig_params):

    atoms_to_keep = get_atom_inds(res_index,simulation)

    for force in zeroed_system.getForces():

        if isinstance(force, HarmonicBondForce):
            force_param_type = 'bond_force_params'
            for bondIndex in xrange(force.getNumBonds()):
                a1,a2,bond_length,bond_k = force.getBondParameters(bondIndex)
                check = [x for x in atoms_to_keep if x in [a1,a2]]

                # if both atoms in res, keep bond energy
                # if only one atom in res, only keep if atom index in res is the smaller of the two
                if (len(check) == 2) or ( (len(check) == 1) and (check[0] == min(a1,a2)) ):
                    force.setBondParameters(bondIndex,*orig_params[force_param_type][bondIndex])

        elif isinstance(force, HarmonicAngleForce):
            force_param_type = 'angle_force_params'
            for angleIndex in xrange(force.getNumAngles()):
                a1,a2,a3,equilib_angle,angle_k = force.getAngleParameters(angleIndex)
                check = [x for x in atoms_to_keep if x in [a1,a2,a3]]

                # only use angle force if more than half the atoms in this res
                if (len(check) >= 2):
                    force.setAngleParameters(angleIndex,*orig_params[force_param_type][angleIndex])

        elif isinstance(force, PeriodicTorsionForce):
            force_param_type = 'torsion_force_params'
            for torsionIndex in xrange(force.getNumTorsions()):
                a1,a2,a3,a4,tors_period,tors_phase,tors_k = force.getTorsionParameters(torsionIndex)
                check = [x for x in atoms_to_keep if x in [a1,a2,a3,a4]]

                # if most of atoms in dihedral in res, use energy
                # if tie, use if min index in res is min index in dihedral
                if (len(check) >= 3) or ( (len(check) == 2) and (min(check) == min(a1,a2,a3,a4)) ):
                    force.setTorsionParameters(torsionIndex,*orig_params[force_param_type][torsionIndex])

        elif isinstance(force, NonbondedForce):
            force_param_type = 'nonbonded_force_params'
            for particleIndex in atoms_to_keep:
                force.setParticleParameters(particleIndex,*orig_params[force_param_type][particleIndex])

            force_param_type = 'nonbonded_exceptions_params'
            for exceptionIndex in xrange(force.getNumExceptions()):
                a1,a2,chargeProd,sigma,epsilon = force.getExceptionParameters(exceptionIndex)
                check = [x for x in atoms_to_keep if x in [a1,a2]]

                # if both atoms in res, use exception energy
                # if only one atom in res, only keep if atom index in res is the smaller of the two
                if (len(check) == 2) or ( (len(check) == 1) and (check[0] == min(a1,a2)) ):
                    force.setExceptionParameters(exceptionIndex,*orig_params[force_param_type][exceptionIndex])

        else:
            raise ValueError("{0} is not a force that is handled in this function yet.".format(force))

        force.updateParametersInContext(simulation.context)


# TODO: make a dict of atom indices and store the bonds, angles, dihedrals, etc
# that they're in, so only have to iterate over those bonds angles etc ?

# node potentials need bonded and nonbonded forces -- edge potentials only need nonbonded
def turn_on_edge(res_i,res_j,simulation,zeroed_system,orig_params):

    atoms_to_keep = get_atom_inds_gen([res_i,res_j],simulation)

    for force in zeroed_system.getForces():

        if isinstance(force, HarmonicBondForce):
            force_param_type = 'bond_force_params'
            for bondIndex in xrange(force.getNumBonds()):
                a1,a2,bond_length,bond_k = force.getBondParameters(bondIndex)
                check = [x for x in atoms_to_keep if x in [a1,a2]]

                # if both atoms in res, keep bond energy
                # if only one atom in res, only keep if atom index in res is the smaller of the two
                if (len(check) == 2) or ( (len(check) == 1) and (check[0] == min(a1,a2)) ):
                    force.setBondParameters(bondIndex,*orig_params[force_param_type][bondIndex])

        elif isinstance(force, HarmonicAngleForce):
            force_param_type = 'angle_force_params'
            for angleIndex in xrange(force.getNumAngles()):
                a1,a2,a3,equilib_angle,angle_k = force.getAngleParameters(angleIndex)
                check = [x for x in atoms_to_keep if x in [a1,a2,a3]]

                # only use angle force if more than half the atoms in this res
                if (len(check) >= 2):
                    force.setAngleParameters(angleIndex,*orig_params[force_param_type][angleIndex])

        elif isinstance(force, PeriodicTorsionForce):
            force_param_type = 'torsion_force_params'
            for torsionIndex in xrange(force.getNumTorsions()):
                a1,a2,a3,a4,tors_period,tors_phase,tors_k = force.getTorsionParameters(torsionIndex)
                check = [x for x in atoms_to_keep if x in [a1,a2,a3,a4]]

                # if most of atoms in dihedral in res, use energy
                # if tie, use if min index in res is min index in dihedral
                if (len(check) >= 3) or ( (len(check) == 2) and (min(check) == min(a1,a2,a3,a4)) ):
                    force.setTorsionParameters(torsionIndex,*orig_params[force_param_type][torsionIndex])

        elif isinstance(force, NonbondedForce):
            force_param_type = 'nonbonded_force_params'
            for particleIndex in atoms_to_keep:
                force.setParticleParameters(particleIndex,*orig_params[force_param_type][particleIndex])

            force_param_type = 'nonbonded_exceptions_params'
            for exceptionIndex in xrange(force.getNumExceptions()):
                a1,a2,chargeProd,sigma,epsilon = force.getExceptionParameters(exceptionIndex)
                check = [x for x in atoms_to_keep if x in [a1,a2]]

                # if both atoms in res, use exception energy
                # if only one atom in res, only keep if atom index in res is the smaller of the two
                if (len(check) == 2) or ( (len(check) == 1) and (check[0] == min(a1,a2)) ):
                    force.setExceptionParameters(exceptionIndex,*orig_params[force_param_type][exceptionIndex])

        else:
            raise ValueError("{0} is not a force that is handled in this function yet.".format(force))

        force.updateParametersInContext(simulation.context)


# these are way slow and should only be used for checking the grid-based funcitons below
def get_system_energy(simulation):
    return simulation.context.getState(getEnergy=True).getPotentialEnergy()

# def get_node_energy(res_index,simulation,system,orig_params):
#     zero_all_forces(system,simulation)
#     turn_on_node(res_index,simulation,system,orig_params)
#     E_node = simulation.context.getState(getEnergy=True).getPotentialEnergy()
#     return E_node
#
# def get_edge_energy(res_i,res_j,simulation,system,orig_params):
#     zero_all_forces(system,simulation)
#     turn_on_edge(res_i,res_j,simulation,system,orig_params)
#     E_both = simulation.context.getState(getEnergy=True).getPotentialEnergy()
#     E_res_i = get_node_energy(res_i,simulation,system,orig_params)
#     E_res_j = get_node_energy(res_j,simulation,system,orig_params)
#     E_edge = E_both - (E_res_i+E_res_j)
#     return E_edge


# # these are way faster than the functions above
# # no passing of intividual grid anymore, just sample denisty,, and construct grid on the fly
# def get_node_energies(res_index,simulation,system,orig_params,N_chi_samples,N_h_chi_samples,res_select=None,h_chi_symmetry=3):
#     """
#     inputs:
#         res_index: (int) zero-indexed residue for this node
#         simulation: the simulation object
#         system: the mutable system object
#         orig_params: original parameters for the system before any node/edges
#             are computed by zeroing out many of the parameters
#         N_chi_samples: samples per chi dof
#         N_h_chi_samples: samples per h_chi dof
#         h_chi_symmetry: symmetry for h_chis (defaults to three, bcz methyl groups)
#     outputs:
#         E_node: (numpy array, float) list of energies correspoding to node states
#     """
#     zero_all_forces(system,simulation)
#     turn_on_node(res_index,simulation,system,orig_params)
#
#     res_type = get_res_name(res_index,simulation)
#
#     node_states = get_total_chi_grid(N_chi_samples, N_h_chi_samples, res_index, simulation, res_select=res_select, h_chi_symmetry=h_chi_symmetry)
#
#     N_states = len(node_states)
#     E_node = Quantity(np.zeros(N_states), unit=kilojoule/mole)
#
#     for state_ind,node_state in enumerate(node_states):
#         (chi_angle_tuple,h_chi_angle_tuple) = node_state
#         set_all_chis_and_h_chis(chi_angle_tuple,h_chi_angle_tuple,res_index,simulation,res_select=res_select,all_positions=None)
#         energy_of_this_state = simulation.context.getState(getEnergy=True).getPotentialEnergy()
#         E_node[state_ind] = energy_of_this_state
#
#     return res_index,E_node
#
# def get_edge_energies(edge,all_node_energies,simulation,system,orig_params,N_chi_samples,N_h_chi_samples,res_select=None,h_chi_symmetry=3):
#     """
#     inputs:
#     res_i,node_i_states,node_i_energies,res_j,node_j_states,node_j_energies
#         res_i/j: (int) zero-indexed residue for these node_i_states
#         all_node_energies: dict of energies tables for all nodes
#         simulation: the simulation object
#         system: the mutable system object
#         orig_params: original parameters for the system before any node/edges
#             are computed by zeroing out many of the parameters
#         N_chi_samples: samples per chi dof
#         N_h_chi_samples: samples per h_chi dof
#         h_chi_symmetry: symmetry for h_chis (defaults to three, bcz methyl groups)
#     outputs:
#         E_node: (numpy array, float) list of energies correspoding to node states
#     """
#
#     # always want i < j in my code
#     res_i,res_j = edge
#     if res_i > res_j:
#         res_j,res_i = res_i,res_j
#         edge = res_i,res_j
#
#     zero_all_forces(system,simulation)
#     turn_on_edge(res_i,res_j,simulation,system,orig_params)
#
#     node_i_states = get_total_chi_grid(N_chi_samples, N_h_chi_samples, res_i, simulation, res_select=res_select, h_chi_symmetry=h_chi_symmetry)
#     node_j_states = get_total_chi_grid(N_chi_samples, N_h_chi_samples, res_j, simulation, res_select=res_select, h_chi_symmetry=h_chi_symmetry)
#
#     node_i_energies = all_node_energies[res_i]
#     node_j_energies = all_node_energies[res_j]
#
#     E_edge = Quantity(np.zeros([len(node_i_energies),len(node_j_energies)]), unit=kilojoule/mole)
#
#     for state_i_ind,(chi_angle_tuple_i,h_chi_angle_tuple_i) in enumerate(node_i_states):
#         set_all_chis_and_h_chis(chi_angle_tuple_i,h_chi_angle_tuple_i,res_i,simulation,res_select=res_select,all_positions=None)
#
#         for state_j_ind,(chi_angle_tuple_j,h_chi_angle_tuple_j) in enumerate(node_j_states):
#             set_all_chis_and_h_chis(chi_angle_tuple_j,h_chi_angle_tuple_j,res_j,simulation,res_select=res_select,all_positions=None)
#
#             E_both = simulation.context.getState(getEnergy=True).getPotentialEnergy()
#             E_edge[state_i_ind,state_j_ind] = E_both - (node_i_energies[state_i_ind] + node_j_energies[state_j_ind])
#
#     return edge,E_edge


# rewrite of the above, with no picklable object getting passed as args
# this means re-making the system from the pdb file and setting params/positions
# which is slow, but maybe worth it if it buys me parallelization
def get_node_energies_frompdbfile(pdb_file,res_index,orig_params,N_chi_samples,N_h_chi_samples,res_select=None,h_chi_symmetry=3):
    """
    inputs:
        pdb_file: (string) pdb file
        res_index: (int) zero-indexed residue for this node
        orig_params: original parameters for the system before any node/edges
            are computed by zeroing out many of the parameters
        N_chi_samples: samples per chi dof
        N_h_chi_samples: samples per h_chi dof
        h_chi_symmetry: symmetry for h_chis (defaults to three, bcz methyl groups)
    outputs:
        E_node: (numpy array, float) list of energies correspoding to node states
    """

    # gotta do this every time since swig objects can't get passed as args to a funciton
    # if we want to parallize that function with pool.map
    pdb = PDBFile(pdb_file)
    forcefield = ForceField('amber99sb.xml')
    system = forcefield.createSystem(pdb.topology,removeCMMotion=False)
    integrator = LangevinIntegrator(298, 1/picosecond, 0.002*picoseconds) # temp shouldn't matter, nothing ever gets integrated
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    turn_on_dielectric(system,simulation)

    zero_all_forces(system,simulation)
    turn_on_node(res_index,simulation,system,orig_params)

    res_type = get_res_name(res_index,simulation)

    node_states = get_total_chi_grid(N_chi_samples, N_h_chi_samples, res_index, simulation, res_select=res_select, h_chi_symmetry=h_chi_symmetry)

    N_states = len(node_states)
    E_node = Quantity(np.zeros(N_states), unit=kilojoule/mole)

    for state_ind,node_state in enumerate(node_states):
        (chi_angle_tuple,h_chi_angle_tuple) = node_state
        set_all_chis_and_h_chis(chi_angle_tuple,h_chi_angle_tuple,res_index,simulation,res_select=res_select,all_positions=None)
        energy_of_this_state = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        E_node[state_ind] = energy_of_this_state

    return res_index,E_node

def get_edge_energies_frompdb(pdb_file,edge,all_node_energies,orig_params,N_chi_samples,N_h_chi_samples,res_select=None,h_chi_symmetry=3):
    """
    inputs:
    res_i,node_i_states,node_i_energies,res_j,node_j_states,node_j_energies
        pdb_file: (string) pdb file
        res_i/j: (int) zero-indexed residue for these node_i_states
        all_node_energies: dict of energies tables for all nodes
        orig_params: original parameters for the system before any node/edges
            are computed by zeroing out many of the parameters
        N_chi_samples: samples per chi dof
        N_h_chi_samples: samples per h_chi dof
        h_chi_symmetry: symmetry for h_chis (defaults to three, bcz methyl groups)
    outputs:
        E_node: (numpy array, float) list of energies correspoding to node states
    """

    # always want i < j in my code
    res_i,res_j = edge
    if res_i > res_j:
        res_j,res_i = res_i,res_j
        edge = res_i,res_j

    # gotta do this every time since swig objects can't get passed as args to a funciton
    # if we want to parallize that function with pool.map
    pdb = PDBFile(pdb_file)
    forcefield = ForceField('amber99sb.xml')
    system = forcefield.createSystem(pdb.topology,removeCMMotion=False)
    integrator = LangevinIntegrator(298, 1/picosecond, 0.002*picoseconds) # temp shouldn't matter, nothing ever gets integrated
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    turn_on_dielectric(system,simulation)

    zero_all_forces(system,simulation)
    turn_on_edge(res_i,res_j,simulation,system,orig_params)

    node_i_states = get_total_chi_grid(N_chi_samples, N_h_chi_samples, res_i, simulation, res_select=res_select, h_chi_symmetry=h_chi_symmetry)
    node_j_states = get_total_chi_grid(N_chi_samples, N_h_chi_samples, res_j, simulation, res_select=res_select, h_chi_symmetry=h_chi_symmetry)

    node_i_energies = all_node_energies[res_i]
    node_j_energies = all_node_energies[res_j]

    E_edge = Quantity(np.zeros([len(node_i_energies),len(node_j_energies)]), unit=kilojoule/mole)

    for state_i_ind,(chi_angle_tuple_i,h_chi_angle_tuple_i) in enumerate(node_i_states):
        set_all_chis_and_h_chis(chi_angle_tuple_i,h_chi_angle_tuple_i,res_i,simulation,res_select=res_select,all_positions=None)

        for state_j_ind,(chi_angle_tuple_j,h_chi_angle_tuple_j) in enumerate(node_j_states):
            set_all_chis_and_h_chis(chi_angle_tuple_j,h_chi_angle_tuple_j,res_j,simulation,res_select=res_select,all_positions=None)

            E_both = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            E_edge[state_i_ind,state_j_ind] = E_both - (node_i_energies[state_i_ind] + node_j_energies[state_j_ind])

    return edge,E_edge


# versions of the above functions where the args are all passed as one tuple
# so that the parallelized map can iterate over the function,
# since a tuple is only one arguement

def get_node_energies_frompdbfile_tup(arg_tuple):
    return get_node_energies_frompdbfile(*arg_tuple)

def get_edge_energies_frompdb_tup(arg_tuple):
    return get_edge_energies_frompdb(*arg_tuple)


# maybe create top level function that takes one arg as a tuple of all args, then iterate through tuple in map?

# Multithreading doesn't work when nested in a function :(
# but the map framework should make it easier down the line

# python functions defined at not the top level can't be pickled.
# pathos & dill should help but also choke on SWIG objects so don't solve the issue.

# threads should be None, or an int, the number of cores you want to run on

# need partial functions for iterating with map

def clone_sys_and_sim(pdb,positions,T=298,ff='amber99sb.xml'):
    forcefield = ForceField(ff)
    system = forcefield.createSystem(pdb.topology,removeCMMotion=False)
    integrator = LangevinIntegrator(T*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(positions)
    return (system,simulation)

# returns a dict of node energies, key = node index, value = energy table
def get_all_node_energies(pdb_file,node_indices,N_chi_samples,N_h_chi_samples,pdb,simulation,system,orig_posits,orig_params,res_select=None,threads=None):

    # temp_sys,temp_sim = clone_sys_and_sim(pdb,orig_posits)

    # get_node_energies_partial = functools.partial(get_node_energies,
    #                                                 simulation=temp_sim,
    #                                                 system=temp_sys,
    #                                                 orig_params=orig_params,
    #                                                 N_chi_samples=N_chi_samples,
    #                                                 N_h_chi_samples=N_h_chi_samples,
    #                                                 res_select=res_select)


    node_indices_and_other_arg_tuples = [(pdb_file,res_index,orig_params,N_chi_samples,N_h_chi_samples,res_select,3) for res_index in node_indices]

    # threads should be None, until pickling bug is fixed
    if threads is not None:
        pool = Pool(threads)
        # all_node_energies = pool.map(get_node_energies_partial,node_indices)
        all_node_energies = pool.map(get_node_energies_frompdbfile_tup,node_indices_and_other_arg_tuples)
        pool.close()
        pool.join()
    else:
        # all_node_energies = map(get_node_energies_partial,node_indices)
        all_node_energies = map(get_node_energies_frompdbfile_tup,node_indices_and_other_arg_tuples)

    all_node_energies_dict = dict((node, energies) for (node, energies) in all_node_energies)

    return all_node_energies_dict


def get_all_edge_energies(pdb_file,edge_indices,all_node_energies,N_chi_samples,N_h_chi_samples,pdb,simulation,system,orig_posits,orig_params,res_select=None,threads=None):

    # temp_sys,temp_sim = clone_sys_and_sim(pdb,orig_posits)

    # get_edge_energies_partial = functools.partial(get_edge_energies,
    #                                                 all_node_energies=all_node_energies,
    #                                                 simulation=temp_sim,
    #                                                 system=temp_sys,
    #                                                 orig_params=orig_params,
    #                                                 N_chi_samples=N_chi_samples,
    #                                                 N_h_chi_samples=N_h_chi_samples,
    #                                                 res_select=res_select)

    edge_indices_and_other_arg_tuples = [(pdb_file,edge,all_node_energies,orig_params,N_chi_samples,N_h_chi_samples,res_select,3) for edge in edge_indices]

    if threads is not None:
        pool = Pool(threads)
        # all_edge_energies = pool.map(get_edge_energies_partial,edge_indices)
        all_edge_energies = pool.map(get_edge_energies_frompdb_tup,edge_indices_and_other_arg_tuples)
        pool.close()
        pool.join()
    else:
        # all_edge_energies = map(get_edge_energies_partial,edge_indices)
        all_edge_energies = map(get_edge_energies_frompdb_tup,edge_indices_and_other_arg_tuples)

    all_edge_energies_dict = dict((edge, energies) for (edge, energies) in all_edge_energies)

    return all_edge_energies_dict


def compute_beta(temp_in_kelvin):
    """
    returns the value of beta for computing boltzmann factors in OpenMM
    openmm energies are in kJ/mole, so avagodro's number is included in beta
        to get reasonable numerics
    intput:
        (float) temperature in degees kelvin
    output:
        (float) beta factor

    """
    if temp_in_kelvin < 0:
        raise ValueError("temperature is less than absolute zero.")
    elif temp_in_kelvin == 0:
        return np.inf
    else:
        T = Quantity(temp_in_kelvin, unit=kelvin)
        NkBT = AVOGADRO_CONSTANT_NA*BOLTZMANN_CONSTANT_kB*T
        return 1/NkBT



# get the indices of the CA atoms in the structure (use C for ACE/NME)
def get_CA_ind(res_index,pdb):
    if get_res_name(res_index,pdb) not in ['ACE','NME']:
        res_atoms_inds_names = get_atom_inds_names(res_index,pdb)
        CA_ind = [ind for (ind,name) in res_atoms_inds_names if name == 'CA']
        return CA_ind[0]
    else:
        res_atoms_inds_names = get_atom_inds_names(res_index,pdb)
        C_ind = [ind for (ind,name) in res_atoms_inds_names if name == 'C']
        return C_ind[0]

# make a graph based on a cut-off distance
def make_cutoff_graph(pdb,positions=None,cutoff=Quantity(value=1.0,unit=nanometer)):
    if positions is None:
        positions = my_simulation.context.getState(getPositions=True).getPositions()

    all_CA_inds = [get_CA_ind(r.index,pdb) for r in pdb.topology.residues()]
    CA_positions = [positions[CA_ind] for CA_ind in all_CA_inds]

    res_indices = [i for (i,obj) in enumerate(pdb.topology.residues())]
    g = G=nx.Graph()
    g.add_nodes_from(res_indices)

    for r in pdb.topology.residues():
        g.node[r.index]['CA_positions'] = CA_positions[r.index]
        g.node[r.index]['name'] = r.name + str(r.index)

    for res_i in res_indices:
        for res_j in res_indices[:res_i]:
            disp = CA_positions[res_i] - CA_positions[res_j]
            dist = np.sqrt(sum([d*d for d in disp]))
            if dist < cutoff:
                g.add_edge(res_i,res_j)

    return g


############################################
# main fuction to construct graph from pdb #
############################################

def make_graph(pdb_file, N_chi_samples, N_h_chi_samples, res_select=None, cutoff=Quantity(value=1.0,unit=nanometer), beta=compute_beta(298), debug=False, threads=None):
    """
    make_graph takes a pdb file and constructs a markov random field
    encoding the single and paiwise energetic interations of each residue in the pdb.
    inputs:
        pdb_file: (file) any pdb file comppatible with openmm.  assumed to not need hydrogens added etc.
        N_chi_samples: samples per chi dof
        N_h_chi_samples: samples per h_chi dof
        res_select: if not None, a list of residue indices to make nodes of, other residues only have one state (ie static)
        cutoff: (quantity) how close do nodes need to be to have and edge between them?
        beta: (quantity) the inverse temperature, in units of moles per energy
        debug: (bool) whether to print out debug messages
    outputs:

    """
    # import pdb structure and construct a simulation
    pdb = PDBFile(pdb_file)
    forcefield = ForceField('amber99sb.xml')
    system = forcefield.createSystem(pdb.topology,removeCMMotion=False)
    integrator = LangevinIntegrator(298, 1/picosecond, 0.002*picoseconds) # temp shouldn't matter, nothing ever gets integrated
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    turn_on_dielectric(system,simulation)

    # simulation.minimizeEnergy()
    ref_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    if debug: print(ref_energy)
    original_parameters = get_all_force_parameters(system)
    original_positions = simulation.context.getState(getPositions=True).getPositions()

    res_indices = [i for (i,obj) in enumerate(pdb.topology.residues())]
    if debug: print(res_indices)
    res_names = [res.name for res in pdb.topology.residues()]
    if debug: print(res_names)


    # initialize graph and add nodes and edges based on cut-off distance
    graph_name = make_cutoff_graph(pdb,positions=original_positions,cutoff=cutoff)

    # get list of edges for later
    edges = [edge for edge in graph_name.edges_iter()]



    # add global attributes
    graph_name.graph['name'] = str(pdb_file)
    graph_name.graph['beta'] = beta
    graph_name.graph['units'] = str(simulation.context.getState(getEnergy=True).getPotentialEnergy()).split(' ')[1]
    graph_name.graph['reference_energy'] = ref_energy


    all_chi_dofs = [get_num_chis_and_hchis(i,pdb,res_select=res_select) for i in res_indices]

    num_chis_samples_per_res = [(N_chi_samples**n_chi)*(N_h_chi_samples**n_hchi) for (n_chi,n_hchi) in all_chi_dofs]

    graph_name.graph['num_chis'] = map(sum, zip(*all_chi_dofs))
    graph_name.graph['grid_points_per_chi'] = (N_chi_samples,N_h_chi_samples)

    # add indices of node state -- seems like there should be a slick way to avoid this but can't figure it out
    for n in graph_name.nodes_iter():
        graph_name.node[n]['node_state_indices'] = [i for i in xrange(num_chis_samples_per_res[n])]


    # get node energies
    all_node_energies = get_all_node_energies(pdb_file,
                                                    res_indices,
                                                    N_chi_samples,
                                                    N_h_chi_samples,
                                                    pdb,
                                                    simulation,
                                                    system,
                                                    original_positions,
                                                    original_parameters,
                                                    res_select=res_select,
                                                    threads=threads
                                                    )

    # add node potentials to graph
    for res_i in graph_name.nodes_iter():
        graph_name.node[res_i]['res_type'] = res_names[res_i]

        node_energies = all_node_energies[res_i]
        node_potential = np.exp(-beta*node_energies).reshape(len(node_energies),1)

        if debug:
            print('node_potential_{0}_shape ='.format(res_i), node_potential.shape)
            #print('node_potential_{0} ='.format(i), node_potential)

        graph_name.node[res_i]['node_potential'] = node_potential


    # get edge energies
    all_edge_energies = get_all_edge_energies(pdb_file,
                                            edges,
                                            all_node_energies,
                                            N_chi_samples,
                                            N_h_chi_samples,
                                            pdb,
                                            simulation,
                                            system,
                                            original_positions,
                                            original_parameters,
                                            res_select=res_select,
                                            threads=threads)


    # add edge potentials to graph
    for edge in graph_name.edges_iter():
        res_i, res_j = edge
        if res_i > res_j:
            res_j, res_i = res_i, res_j
            edge = res_i, res_j

        edge_energies = all_edge_energies[edge]
        edge_potential = np.exp(-beta*edge_energies)
        if debug:
            print('egde_potential_{0}_shape ='.format(str(res_i)+str(res_j)), edge_potential.shape)
        graph_name.edge[res_i][res_j]['edge_potential'] = edge_potential

    return graph_name

# takes a graph with a large cutoff and returns a copy of it with a smaller cutoff
def prune_graph(graph,cutoff=Quantity(value=1.0,unit=nanometer)):

    pruned_graph = graph.copy()

    for edge in graph.edges_iter():
        n_a,n_b = edge
        disp = pruned_graph.node[n_a]['CA_positions'] - pruned_graph.node[n_b]['CA_positions']
        dist = np.sqrt(sum([d*d for d in disp]))
        if dist > cutoff:
            pruned_graph.remove_edge(n_a,n_b)

    return pruned_graph
