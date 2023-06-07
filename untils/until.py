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

def split_data(data,split_case,ratio,cell_names, random_state):
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

def load_CCLE_data(base_path):
    GDSC_rma_path = base_path+"/CCLE/CCLE_Data/CCLE_RNAseq.csv"
    GDSC_variant_path = base_path+"/CCLE/CCLE_Data/CCLE_DepMap.csv"
    GDSC_smiles_path = base_path+"/CCLE/CCLE_Data/CCLE_smiles.csv"

    rma = pd.read_csv(GDSC_rma_path, index_col=0)
    var = pd.read_csv(GDSC_variant_path, index_col=0)
    smiles = pd.read_csv(GDSC_smiles_path, index_col=0)

    return rma, var, smiles


def get_data(data_url, cache_subdir, download=True, svn=False):
    print('downloading data')
    # cache_subdir = os.path.join(CANDLE_DATA_DIR, 'SWnet', 'Data')
    
    if download and svn:
        os.makedirs(cache_subdir, exist_ok=True)
        os.system(f'svn checkout {data_url} {cache_subdir}')   
        print('downloading done') 

    elif download and svn==False:
        os.makedirs(cache_subdir, exist_ok=True)
        ccle_data = os.path.join(cache_subdir,'CCLE/CCLE_Data/')
        os.makedirs(ccle_data, exist_ok=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/zuozhaorui/SWnet/master/data/CCLE/CCLE_Data/CCLE_DepMap.csv',
         f'{ccle_data}/CCLE_DepMap.csv' )