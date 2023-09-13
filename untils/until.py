import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib

def get_drug_dict(smiles):
    try:
        drug_index = smiles.index.astype('float32')
    except:
        drug_index = smiles.index
    drug_dict = {}
    for i in range(len(drug_index)):
        drug_dict[drug_index[i]] = i
    return drug_dict

def split_data(data,split_case,ratio,cell_names, random_state=42):
    data = data[data["labels"].notnull()]
    data = data[~data['drug_id'].isin([185, 1021])]  # except drug id
    # Split data sets randomly
    if split_case == 0:
        train_id, test_id = train_test_split(data, test_size=1-ratio, random_state=random_state)
        val_id, test_id = train_test_split(test_id, test_size=0.5, random_state=random_state)

    # split data sets by cells
    elif split_case == 1:
        np.random.seed(0)
        np.random.shuffle(cell_names)
        n = int(ratio * len(cell_names))
        train_id = data[data['cell_line_id'].isin(cell_names[:n])]
        test_id = data[data['cell_line_id'].isin(cell_names[n:])]
    # all data sets
    elif split_case == 2:
        train_id = data
        _ , test_id = train_test_split(data, test_size=1 - ratio, random_state=0)

    elif split_case == 3:
        train_id, test_id = train_test_split(data, test_size=1 - ratio, random_state=0)
        val_id = test_id
    # train_id, test_id = train_test_split(data, test_size=1 - ratio, random_state=0)
    # train_id, test_id = train_id[0:100], test_id[0:100]
    return train_id, val_id, test_id

def load_GDSC_data(base_path):
    # GDSC_rma_path = base_path+"data/GDSC/GDSC_data/GDSC_rma.csv"   # original
    # GDSC_variant_path = base_path+"data/GDSC/GDSC_data/GDSC_variant.csv"  # original
    # GDSC_smiles_path = base_path+"data/GDSC/GDSC_data/GDSC_smiles.csv" # original
    GDSC_rma_path = base_path+"/GDSC/GDSC_data/GDSC_rma.csv"
    GDSC_variant_path = base_path+"/GDSC/GDSC_data/GDSC_variant.csv"
    GDSC_smiles_path = base_path+"/GDSC/GDSC_data/GDSC_smiles.csv"

    rma = pd.read_csv(GDSC_rma_path, index_col=0)
    var = pd.read_csv(GDSC_variant_path, index_col=0)
    smiles = pd.read_csv(GDSC_smiles_path, index_col=0)

    return rma, var, smiles

def load_CCLE_data(base_path, data_type, cross_study=False):
    GDSC_rma_path = base_path+f"/{data_type}/{data_type}_Data/{data_type}_RNAseq.csv"
    GDSC_variant_path = base_path+f"/{data_type}/{data_type}_Data/{data_type}_DepMap.csv"
    GDSC_smiles_path = base_path+f"/{data_type}/{data_type}_Data/{data_type}_smiles.csv"
    if cross_study:
        all_smiles_path = base_path+f"/{data_type}/{data_type}_Data/all_smiles.csv"
        all_smiles = pd.read_csv(all_smiles_path, index_col=0)
    else:
        all_smiles = None

    rma = pd.read_csv(GDSC_rma_path, index_col=0)
    var = pd.read_csv(GDSC_variant_path, index_col=0)
    smiles = pd.read_csv(GDSC_smiles_path, index_col=0)

    return rma, var, smiles, all_smiles


def get_data(data_url, cache_subdir, radius=3, download=True, svn=False):
    # cache_subdir = os.path.join(CANDLE_DATA_DIR, 'SWnet', 'Data')
    
    if download and svn:
        os.makedirs(cache_subdir, exist_ok=True)
        os.system(f'svn checkout {data_url} {cache_subdir}')   
        print('downloading done') 

    elif download and svn==False:
        # os.makedirs(cache_subdir, exist_ok=True)
        ccle_data = os.path.join(cache_subdir,'CCLE/CCLE_Data/')
        os.makedirs(ccle_data, exist_ok=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/CCLE_Data/CCLE_DepMap.csv',
         f'{ccle_data}/CCLE_DepMap.csv')
        urllib.request.urlretrieve('https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/CCLE_Data/CCLE_RNAseq.csv',
         f'{ccle_data}/CCLE_RNAseq.csv')
        urllib.request.urlretrieve('https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/CCLE_Data/CCLE_cell_drug_labels.csv',
         f'{ccle_data}/CCLE_cell_drug_labels.csv')
        urllib.request.urlretrieve('https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/CCLE_Data/CCLE_smiles.csv',
         f'{ccle_data}/CCLE_smiles.csv')

        ccle_data = os.path.join(cache_subdir,'CCLE/drug_similarity/')
        os.makedirs(ccle_data, exist_ok=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/drug_similarity/CCLE_drug_similarity.csv',
         f'{ccle_data}/CCLE_drug_similarity.csv')
         
        ccle_data = os.path.join(cache_subdir,f'CCLE/graph_data/radius{radius}')
        os.makedirs(ccle_data, exist_ok=True)
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/graph_data/radius{radius}/Smiles.txt',
         f'{ccle_data}/Smiles.txt')
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/graph_data/radius{radius}/adjacencies.npy',
         f'{ccle_data}/adjacencies.npy')
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/graph_data/radius{radius}/compounds.npy',
         f'{ccle_data}/compounds.npy')
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/graph_data/radius{radius}/fingerprint_dict.pickle',
         f'{ccle_data}/fingerprint_dict.pickle')

         
