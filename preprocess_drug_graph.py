from collections import defaultdict
import os
import pickle
import sys
import argparse

import numpy as np
import pandas as pd

from rdkit import Chem
from data_utils import candle_data_dict
from process import candle_preprocess

from data_utils import Downloader, DataProcessor, add_smiles, remove_smiles_with_noneighbor_frags
CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")

def create_data_dirs(data_path, data_type):

    for p in [f'{data_type}/{data_type}_Data', f'{data_type}/drug_similarity', f'{data_type}/graph_data']:
        path = os.path.join(data_path, p)
        if not os.path.exists(path):
            os.makedirs(path)

def get_data_type_data(data_type, data_path, opt):

    metric = opt['metric']
    data_processor = DataProcessor(opt['data_version'])
    split_id = opt['data_split_id']

    rs_train = data_processor.load_drug_response_data(data_path, data_type=data_type, split_id=split_id, split_type='train', response_type=metric)
    rs_val = data_processor.load_drug_response_data(data_path, data_type=data_type, split_id=split_id, split_type='val', response_type=metric)
    rs_test = data_processor.load_drug_response_data(data_path, data_type=data_type, split_id=split_id, split_type='test', response_type=metric)

    smiles_df = data_processor.load_smiles_data(data_path)
    smiles_df = remove_smiles_with_noneighbor_frags(smiles_df)

    train_df = add_smiles(smiles_df,rs_train, metric)
    val_df = add_smiles(smiles_df,rs_val, metric)
    test_df = add_smiles(smiles_df,rs_test, metric)

    df1 = pd.concat([train_df, val_df,test_df], axis=0)
    return df1


# if __name__ == "__main__":
def prepare_graph_data(data_path, opt):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--radius" ,type=int)
    # parser.add_argument("--data_path" ,type=str)
    # args = parser.parse_args()
    radius = opt['radius']
    data_type = opt['data_type']
    other_ds = opt['other_ds']
    data_version = opt['data_version']
    radius = opt['radius']

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
            other_ds = [i.strip() for i in  opt['other_ds'].split(',')]
            other_ds2 = [data_type] + [candle_data_dict[ds] for ds in other_ds]
            print('other ds: ', other_ds)
            data_split_id = opt['data_split_id']
            data_path=os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data')
            downloader = Downloader(data_version)   
            dfs=[]         
            for ds in other_ds2:

                data_type_other = ds
                create_data_dirs(data_path, data_type_other)
                downloader.download_candle_data(data_type=data_type_other, split_id=data_split_id, data_dest=data_path)

                ext_gene_file = os.path.join(data_path, 'swn_original','CCLE/CCLE_Data/CCLE_DepMap.csv')
  
                candle_preprocess(data_type=data_type_other, 
                                    metric=opt['metric'], 
                                    data_path=data_path,
                                    split_id=data_split_id,
                                    ext_gene_file=ext_gene_file,
                                    params=opt)
                dfs.append(get_data_type_data(data_type_other, data_path, opt))

            dfs = pd.concat(dfs, axis=0)
            dfs.drop_duplicates(subset=['smiles'], inplace=True)
            dfs.reset_index(drop=True, inplace=True)
            dfs = dfs[['improve_chem_id', 'smiles']]
            dfs.set_index('improve_chem_id', inplace=True)
            
            filename = data_path+f'/{data_type}/{data_type}_Data/all_smiles2.csv' # changing this for comatibility with all the data sources, have to find a better fix
            dfs.to_csv(filename)

            for ds in other_ds:
                print('creating ln')
                dsn = candle_data_dict[ds]
                os.system(f'ln -s ../../{data_type}/{data_type}_Data/all_smiles2.csv {data_path}/{dsn}/{dsn}_Data/all_smiles2.csv')
                os.system(f'ln -s ../../{data_type}/graph_data/radius{radius} {data_path}/{dsn}/graph_data/radius{radius}')
                os.system(f'ln -s ../../{data_type}/drug_similarity/{data_type}_drug_similarity.csv {data_path}/{dsn}/drug_similarity/{dsn}_drug_similarity.csv')
                


    else:
        filename = data_path+f'/{data_type}/{data_type}_Data/{data_type}_smiles.csv' # changing this for comatibility with all the data sources, have to find a better fix

    # return None

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
