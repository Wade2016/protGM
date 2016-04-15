# README

This python code is written to be run through a [Jupyter notebook](http://jupyter.org).

The general idea is that you input a `.pdb` file and eventially you get a free energy prediction.

On the way there, you generate a (networkx) graph from the pdb:

 - Nodes on the graph are residues in the pdb, and each contains a potential, which weights the preferences of that node to be in different conformations, according to a given forcefield.
 - Similarly, edges connecting two nodes in the graph contain a matrix weighting the pairwise preferences of the two nodes for the different combinations of conformations they can be in.
- The weights of each entry in the potentials are the Boltzmann factors of the relevant self- or pairwise-interaction energies.  The node potentials are N×1 matrices, and the edge potential are N×M matrices.

## External dependencies
First install [Anaconda Python](https://www.continuum.io/downloads) (the version number will change from 2.5.0):

```
curl -O http://repo.continuum.io/archive/Anaconda2-2.5.0-Linux-x86_64.sh
bash Anaconda2-2.5.0-Linux-x86_64.sh
```

Then install the [Omnia](http://www.omnia.md) suite (OpenMM, PDBFixer, MDTraj, ParmEd, etc):

```
conda config --add channels omnia
conda install omnia
```

Lastly, I use [Seaborn](https://stanford.edu/~mwaskom/software/seaborn/) for plotting:

```
conda install seaborn
```

You might need to install some more packages. Just **make sure to install with conda**:

```
conda install <some_package>
```

**Do not use pip or easy_install** or you risk confusing your dependencies.

## Internal dependencies/structure
- generating energy tables via openmm is done in `openmm_potentials.py`
- the belief propagation code is in `belief_propogation.py`
- the brute force code is in `exact_partition_functions`
- the graph utilites (plotting, edge inspection, etc) are in `graph_utilities.py`

# TODO

## Big/Conceptual Stuff
- come up with good test systems


## Medium Stuff


## Little Stuff
- clean out old chi angle getting and setting code


## "Future Work"
- pruning states that are clashes
- smarter than uniform-grid sampling
  - higher sample density for "earlier" chis?
- backbone flexibility (not enough to change graph topology, but enough to help binding)
-  use something faster than openmm to generate the potentials.  openmm is the bottleneck for speed.
-  if the graph algorithms ever become the bottleneck, look into using libDAI
