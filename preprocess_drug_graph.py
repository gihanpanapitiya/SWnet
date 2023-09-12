from collections import defaultdict
import os
import pickle
import sys
import argparse

import numpy as np
import pandas as pd

from rdkit import Chem





# if __name__ == "__main__":
def prepare_graph_data(data_path, opt):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--radius" ,type=int)
    # parser.add_argument("--data_path" ,type=str)
    # args = parser.parse_args()
    radius = opt['radius']
    data_type = opt['data_type']
    # data_path = opt['data_path']
    print('radius = {}'.format(radius))


    def create_atoms(mol):
        """Create a list of atom (e.g., hydrogen and oxygen) IDs
        considering the aromaticity."""
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [atom_dict[a] for a in atoms]
        return np.array(atoms)


    def create_ijbonddict(mol):
        """Create a dictionary, which each key is a node ID
        and each value is the tuples of its neighboring node
        and bond (e.g., single and double) IDs."""
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict


    def extract_fingerprints(atoms, i_jbond_dict, radius):
        """Extract the r-radius subgraphs (i.e., fingerprints)
        from a molecular graph using Weisfeiler-Lehman algorithm."""

        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                """Update each node ID considering its neighboring nodes and edges
                (i.e., r-radius subgraphs or fingerprints)."""
                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(fingerprint_dict[fingerprint])
                nodes = fingerprints

                """Also update each edge ID considering two nodes
                on its both sides."""
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = edge_dict[(both_side, edge)]
                        _i_jedge_dict[i].append((j, edge))
                i_jedge_dict = _i_jedge_dict

        return np.array(fingerprints)


    def create_adjacency(mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)

    def dump_dictionary(dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(dictionary), f)



    """Load a dataset."""
    print('creating graph data', data_type)
    if opt["cross_study"]:
        filename = data_path+f'/{data_type}/{data_type}_Data/all_smiles.csv' # changing this for comatibility with all the data sources, have to find a better fix
    else:
        filename = data_path+f'/{data_type}/{data_type}_Data/{data_type}_smiles.csv' # changing this for comatibility with all the data sources, have to find a better fix

    print('----------------------')
    print(filename)
    print('----------------------')
    data = pd.read_csv(filename, index_col=0)
    all_smiles = data["smiles"].values

    print("smiles: ", len(all_smiles), all_smiles[:100])

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    Smiles, compounds, adjacencies= '', [], []
    failed_smiles=[]
    # for smiles in all_smiles:
    for id  in data.index:
        smiles = data.loc[id, 'smiles']

        # print(smiles)

        try:
            Smiles += smiles + '\n'

            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
            atoms = create_atoms(mol)
            i_jbond_dict = create_ijbonddict(mol)

            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
            compounds.append(fingerprints)

            adjacency = create_adjacency(mol)
            adjacencies.append(adjacency)
        except:
            failed_smiles.append([id,smiles])




    dir_input = (data_path+f'/{data_type}/graph_data/'+'radius' + str(radius) + '/')
    os.makedirs(dir_input, exist_ok=True)

    with open(dir_input + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    # np.save(dir_input + 'compounds', compounds)
    # np.save(dir_input + 'adjacencies', adjacencies)

    with open(dir_input + 'compounds', 'wb') as f:
        pickle.dump(compounds, f)

    with open(dir_input + 'adjacencies', 'wb') as f:
        pickle.dump(adjacencies, f)
        
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')

    print(f'The preprocess of {data_type} dataset has finished!')
