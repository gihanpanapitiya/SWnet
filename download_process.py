import logging
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
import sys
import pickle
import argparse
import untils.until as untils
from rdkit import Chem
import random
import candle
import json
import logging
from preprocess_drug_graph import prepare_graph_data
from preprocess_drug_similarity import prepare_similarity_data
import urllib
from preprocess import candle_preprocess
from sklearn.model_selection import train_test_split
from data_utils import Downloader, DataProcessor
# from data_utils import download_candle_data
# torch.manual_seed(0)
from preprocess import get_data

file_path = os.path.dirname(os.path.realpath(__file__))
additional_definitions = [
    {'name': 'batch_size',
     'type': int
     },
    {'name': 'lr',
     'type': float,
     'help': 'learning rate'
     },
    {'name': 'epochs',
     'type': int
     },
    {'name': 'step_size',
     'type': int
     },
    {'name': 'gamma',
     'type': float
     },
    {'name': 'radius',
     'type': int
     },
    {'name': 'dim',
     'type': int
     },
    {'name': 'layer_gnn',
     'type': int
     },
    {'name': 'data_type',
     'type': str
     },
    {'name': 'data_url',
     'type': str
     },
    {'name': 'data_split_seed',
     'type': int
     },
     {'name': 'metric',
     'type': str
     },
    {'name': 'download_data',
     'type': bool
     },
    {'name': 'cross_study',
     'type': bool
     },
    {'name': 'data_source',
     'type': str
     }  
]

required = None


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


device_ids = [ int(os.environ["CUDA_VISIBLE_DEVICES"]) ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")


def load_tensor(file_name, dtype):
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
    return [dtype(d).to(device) for d in file]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class CreateDataset(Dataset):
    def __init__(self, rma_all, var_all, id):
        self.rma_all = rma_all
        self.var_all = var_all
        self.all_id = id.values

    def __len__(self):
        return len(self.all_id)

    def __getitem__(self, idx):
        cell_line_id = self.all_id[idx][0]
        drug_id = self.all_id[idx][1]
        y = np.float32(self.all_id[idx][2])

        rma = self.rma_all.loc[cell_line_id].values.astype('float32')
        var = self.var_all.loc[cell_line_id].values.astype('float32')
        return rma, var, drug_id, y



def add_natoms(df):
    natoms = [Chem.MolFromSmiles(i).GetNumAtoms() for i in df.smiles]
    df['natoms'] = natoms



def run(gParameters):

    """hyper-parameter"""

    print(gParameters)
    LR = gParameters['lr']
    BATCH_SIZE = gParameters['batch_size']
    num_epochs = gParameters['epochs']
    step_size = gParameters['step_size']
    gamma = gParameters['gamma']
    split_case = gParameters['split_case']
    radius = gParameters['radius']
    dim = gParameters['dim']
    layer_gnn = gParameters['layer_gnn']
    output_dir = gParameters['output_dir']
    data_url = gParameters['data_url']
    download_data = gParameters['download_data']
    metric = gParameters['metric']
    data_path=os.path.join(CANDLE_DATA_DIR, gParameters['model_name'], 'Data')
    data_source = gParameters['data_source']
    data_split_id = gParameters['data_split_id']
    cross_study = gParameters['cross_study']
    data_version = gParameters['data_version']

    print("BATCH_SIZE: ", BATCH_SIZE)

    """log"""
    dt = datetime.now() 
    file_name = os.path.basename(__file__)[:-3]
    date = dt.strftime('_%Y%m%d_%H_%M_%S')



    if 'original' in data_source:
        n_genes=1478
        # currently works with original CCLE data
        if data_source == 'ccle_original':
            data_type='CCLE'
        elif data_source == 'gdsc_original':
            data_type='GDSC'

        untils.get_data(data_url=data_url, cache_subdir=data_path, radius=radius, download=True, data_type=data_type)
        gParameters['data_type'] = data_type
        prepare_graph_data(data_path, gParameters)
        prepare_similarity_data(data_path, data_type, gParameters)


    # elif gParameters['data_type'] == 'ccle_candle':
    elif 'candle' in data_source:
        if data_source == 'ccle_candle':
            data_type = "CCLE"
        elif data_source == 'ctrpv2_candle':
            data_type = "CTRPv2"
        elif data_source == 'gdscv1_candle':
            data_type = "GDSCv1"
        elif data_source == 'gdscv2_candle':
            data_type = "GDSCv2"
        elif data_source == 'gcsi_candle':
            data_type = "gCSI"

        gParameters['data_type'] = data_type

        st_pp = time.time() 
        print("Creating data for candle" )
        untils.get_data(data_url=data_url, cache_subdir=os.path.join(data_path, 'swn_original'), radius=radius, download=True, svn=False)
        #if download_data:
        #get_data(data_url, os.path.join(data_path, 'swn_original'), True, False)
        downloader = Downloader(data_version)
        downloader.download_candle_data(data_type=data_type, split_id=data_split_id, data_dest=data_path)
        #     download_candle_data(data_type=data_type, split_id=data_split_id, data_dest=data_path)    

        for p in [f'{data_type}/{data_type}_Data', f'{data_type}/drug_similarity', f'{data_type}/graph_data']:
            path = os.path.join(data_path, p)
            if not os.path.exists(path):
                os.makedirs(path)


        ext_gene_file = os.path.join(data_path, 'swn_original','CCLE/CCLE_Data/CCLE_DepMap.csv')
  
        candle_preprocess(data_type=data_type, 
                         metric=metric, 
                         data_path=data_path,
                         split_id=data_split_id,
                         ext_gene_file=ext_gene_file,
                         params=gParameters)

        # df_mut = pd.read_csv(os.path.join(data_path,f'{data_type}/{data_type}_Data', f'{data_type}_RNAseq.csv'), index_col=0)
        # n_genes = len(df_mut.columns)

        if not os.path.exists(os.path.join(data_path, data_type, f'graph_data/radius{radius}/compounds')):
            print("creating graph data")
            prepare_graph_data(data_path, gParameters)
        # os.system(f"python preprocess_drug_graph.py --radius {radius} --data_path {data_path}")
        if not os.path.exists(os.path.join(data_path, data_type, f'drug_similarity/{data_type}_drug_similarity.csv')):
            print("creating similarity data")
            prepare_similarity_data(data_path, data_type, gParameters)
        pp_time = time.time() - st_pp


class SWnet_candle(candle.Benchmark):

        def set_locals(self):
            if required is not None:
                self.required = set(required)
            if additional_definitions is not None:
                self.additional_definitions = additional_definitions

def initialize_parameters():
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    swnet_params = SWnet_candle(
        filepath = file_path,
        defmodel = "swnet_ccle_model.txt",
        framework = "pytorch",
        prog="SWnet",
        desc="CANDLE compliant SWnet",
    )
    gParameters = candle.finalize_parameters(swnet_params)
    return gParameters

if __name__ == '__main__':

    gParameters = initialize_parameters()
    print(gParameters)
    run(gParameters)
    print("Done preprocessing.")

