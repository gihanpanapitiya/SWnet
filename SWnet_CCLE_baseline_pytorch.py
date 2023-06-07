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

# torch.manual_seed(0)
file_path = os.path.dirname(os.path.realpath(__file__))
additional_definitions = [
    {'name': 'batch_size',
     'type': int
     },
    {'name': 'lr',
     'type': float,
     'help': 'learning rate'
     },
    {'name': 'num_epochs',
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
     }
]

required = None


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)


device_ids = [ int(os.environ["CUDA_VISIBLE_DEVICES"]) ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)



# graph dataset
# def load_tensor(file_name, dtype):
#     return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

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


class Model(nn.Module):
    def __init__(self,dim,layer_gnn,drugs_num, n_fingerprint, similarity_softmax,
            GDSC_drug_dict, graph_dataset, n_genes):
        super(Model, self).__init__()
        self.n_genes = n_genes
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(drugs_num, n_genes), requires_grad=True).to(device)
        self.fuse_weight.data.normal_(0.5, 0.25)

        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])

        self.similarity_softmax = similarity_softmax # gihan -> what is this?
        self.GDSC_drug_dict = GDSC_drug_dict
        self.graph_dataset = graph_dataset
        self.dim = dim
        self.layer_gnn = layer_gnn

        self.gene = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=15, stride=2),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            # nn.Conv1d(in_channels=10, out_channels=10, kernel_size=30, stride=2),
            # nn.BatchNorm1d(10),
            # nn.ReLU(),
            nn.Conv1d(in_channels=10, out_channels=5, kernel_size=15, stride=2),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            # nn.Linear(71, 32), # for original
            nn.Linear(70, 32), # for candle ccle
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.merged = nn.Sequential(
            nn.Linear(210,100),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=10, stride=2),
            nn.BatchNorm1d(5),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=10, stride=2),
            nn.BatchNorm1d(5),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=0.1)
        )
        self.out = nn.Sequential(
            # nn.Linear(40, 20),
            # nn.Dropout(p=0.1),
            nn.Linear(5,1)
        )

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention(self,fuse_weight,drug_ids,var):
        try:
            drug_ids = drug_ids.numpy().tolist()
        except:
            drug_ids = drug_ids

        com = torch.zeros(len(drug_ids), self.n_genes).to(device)

        for i in range(len(drug_ids)):
            com[i] = torch.mv(fuse_weight.permute(1, 0), self.similarity_softmax[self.GDSC_drug_dict[drug_ids[i]]]) * var[i]

        return com.view(-1,self.n_genes)

    def combine(self, rma, var,drug_id):
        self.fuse_weight.data = torch.clamp(self.fuse_weight, 0, 1)
        attention_var = self.attention(self.fuse_weight, drug_id, var)
        z = rma + attention_var
        return z

    def forward(self, rma, var, drug_id):
        com = self.combine(rma, var,drug_id)
        com = com.unsqueeze(1)
        out = self.gene(com)
        out_gene = out.view(out.size(0), -1)

        """Compound vector with GNN."""
        batch_graph = [self.graph_dataset[self.GDSC_drug_dict[i]] for i in drug_id]
        compound_vector = torch.FloatTensor(len(drug_id), self.dim).to(device)
        for i, graph in enumerate(batch_graph):
            fingerprints, adjacency = graph
            fingerprints.to(device)
            adjacency.to(device)
            fingerprint_vectors = self.embed_fingerprint(fingerprints)
            compound_vector[i] = self.gnn(fingerprint_vectors, adjacency, self.layer_gnn)

        # print(out.size(0),out.size(1))

        concat = torch.cat([out_gene, compound_vector], dim=1)
        # concat = concat.view(concat.size(0), -1)
        concat = concat.unsqueeze(1)

        merge = self.merged(concat)
        # print(merge.size(0), merge.size(1))
        merge = merge.view(merge.size(0), -1)

        y_pred = self.out(merge)
        return y_pred


def train_model(model, train_loader, test_loader, dataset_sizes, criterion, optimizer, scheduler, output_dir, file_name, num_epochs=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10.0
    best_epoch = -1
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        log.info('Epoch {}/{}\n'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        train_loss = 0.0
        model.train()
        for step, (rma, var, drug_id, y) in tqdm(enumerate(train_loader)):
            rma = rma.cuda(device=device_ids[0])
            var = var.cuda(device=device_ids[0])
            y = y.cuda(device=device_ids[0])
            y = y.view(-1, 1)
            # print('y',y)

            y_pred = model(rma, var, drug_id)
            # print('y_pred',y_pred)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * rma.size(0)
        scheduler.step()

        test_loss = 0.0
        model.eval()
        for step, (rma, var, drug_id, y) in tqdm(enumerate(test_loader)):
            rma = rma.cuda(device=device_ids[0])
            var = var.cuda(device=device_ids[0])
            y = y.cuda(device=device_ids[0])
            y = y.view(-1, 1)

            y_pred = model(rma, var, drug_id)
            loss = criterion(y_pred, y)
            test_loss += loss.item() * rma.size(0)

        epoch_train_loss = train_loss / dataset_sizes['train']
        epoch_test_loss = test_loss / dataset_sizes['test']

        print('Train Loss: {:.4f} Test Loss: {:.4f}'.format(epoch_train_loss, epoch_test_loss))
        log.info('Train Loss: {:.4f} Test Loss: {:.4f}\n'.format(epoch_train_loss, epoch_test_loss))
   
        # deep copy the model
        if epoch_test_loss < best_loss and epoch>=3:
            best_loss = epoch_test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch=epoch
        
    print("best epoch: ", best_epoch)
    time_elapsed = time.time() - since
    print("train time: ", time_elapsed)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test loss: {:4f}'.format(best_loss))
    log.info('Best test loss: {:4f}\n'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)

    

    # pth_name = '../log/pth/' + str(round(best_loss,4)) + '_' + file_name + '_r' + str(radius) +'_s' + str(split_case) + '.pth'
    # torch.save(model.state_dict(), pth_name)
    pth_name = 'best_model.pth' # gihan
    pth_name = os.path.join(output_dir, pth_name)
    torch.save(model.state_dict(), pth_name)
    print("Save model done!")
    return model, best_model_wts


def add_natoms(df):
    natoms = [Chem.MolFromSmiles(i).GetNumAtoms() for i in df.smiles]
    df['natoms'] = natoms

def eval_model(model, test_loader, ccle_smiles):
    from sklearn.metrics import r2_score, mean_squared_error
    y_pred = []
    y_true = []
    smiles = []
    model.eval()
    for step, (rma, var, drug_id,y) in tqdm(enumerate(test_loader)):
        rma = rma.cuda(device=device_ids[0])
        var = var.cuda(device=device_ids[0])
        y = y.cuda(device=device_ids[0])
        y = y.view(-1, 1)
        # print('y',y)
        y_true += y.cpu().detach().numpy().tolist()
        y_pred_step = model(rma, var, drug_id)
        y_pred += y_pred_step.cpu().detach().numpy().tolist()
        smiles.extend( [ccle_smiles.loc[di, 'smiles'] for di in drug_id] )

    df_res = pd.DataFrame(zip(np.array(y_true).ravel(), np.array(y_pred).ravel(), smiles ), columns=['true', 'pred', 'smiles'])
    return mean_squared_error(y_true, y_pred),r2_score(y_true, y_pred), df_res

def download_csa_data(opt):

    csa_data_folder = os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data', 'csa_data', 'raw_data')
    splits_dir = os.path.join(csa_data_folder, 'splits') 
    x_data_dir = os.path.join(csa_data_folder, 'x_data')
    y_data_dir = os.path.join(csa_data_folder, 'y_data')

    if not os.path.exists(csa_data_folder):
        print('creating folder: %s'%csa_data_folder)
        os.makedirs(csa_data_folder)
        os.mkdir( splits_dir  )
        os.mkdir( x_data_dir  )
        os.mkdir( y_data_dir  )
    

    for file in ['CCLE_all.txt', 'CCLE_split_0_test.txt', 'CCLE_split_0_train.txt', 'CCLE_split_0_val.txt']:
        urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data/splits/{file}',
        splits_dir+f'/{file}')

    for file in ['cancer_mutation_count.txt', 'drug_SMILES.txt','drug_ecfp4_512bit.txt', 'cancer_gene_expression.txt']:
        urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data/x_data/{file}',
        x_data_dir+f'/{file}')

    for file in ['response.txt']:
        urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/csa_data/y_data/{file}',
        y_data_dir+f'/{file}')


def run(gParameters):

    """hyper-parameter"""

    print(gParameters)
    LR = gParameters['lr']
    BATCH_SIZE = gParameters['batch_size']
    num_epochs = gParameters['num_epochs']
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


    """log"""
    dt = datetime.now()  # 创建一个datetime类对象
    file_name = os.path.basename(__file__)[:-3]
    date = dt.strftime('_%Y%m%d_%H_%M_%S')

    log.info(file_name + date + '.csv \n')
    log.info('radius = {:d},split case = {:d}\n'.format(radius, split_case))
    print("Log is start!")

    if gParameters['data_type'] == 'original':

        untils.get_data(data_url, data_path, download_data)
        n_genes=1478



    elif gParameters['data_type'] == 'ccle_candle':
        st_pp = time.time() 
        # print("Creating data for candle")
        untils.get_data(data_url, os.path.join(data_path, 'swn_original'), True, False)
    
        if not os.path.exists(data_path+'/csa_data'):
            download_csa_data(gParameters)
        
        # for p in ['CCLE/CCLE_Data', 'CCLE/CCLE_Data/drug_similarity', 'CCLE/CCLE_Data/graph_data', f'CCLE/CCLE_Data/graph_data/radius_{radius}']:
        for p in ['CCLE/CCLE_Data', 'CCLE/drug_similarity', 'CCLE/graph_data']:
            path = os.path.join(data_path, p)
            if not os.path.exists(path):
                os.makedirs(path)
            
        # smi_save_path = os.path.join(data_path,'CCLE/CCLE_Data', 'CCLE_smiles.csv')
        # expr_save_path = os.path.join(data_path,'CCLE/CCLE_Data', 'CCLE_RNAseq.csv')
        # mut_save_path = os.path.join(data_path,'CCLE/CCLE_Data', 'CCLE_DepMap.csv')
        # data_save_path = os.path.join(data_path,'CCLE/CCLE_Data', 'CCLE_cell_drug_labels.csv')
        # ext_gene_file = os.path.join(data_path,'CCLE/CCLE_data', 'CCLE_cell_drug_labels.csv')
        ext_gene_file = os.path.join(data_path, 'swn_original','CCLE/CCLE_Data/CCLE_DepMap.csv')
  
        candle_preprocess(data_type='CCLE', 
                         metric=metric, 
                         data_path=data_path,
                         ext_gene_file=ext_gene_file)

        df_mut = pd.read_csv(os.path.join(data_path,'CCLE/CCLE_Data', 'CCLE_RNAseq.csv'), index_col=0)
        n_genes = len(df_mut.columns)


        prepare_graph_data(data_path, gParameters)
        # os.system(f"python preprocess_drug_graph.py --radius {radius} --data_path {data_path}")
        prepare_similarity_data(data_path)
        pp_time = time.time() - st_pp

    """Load preprocessed drug graph data."""
    dir_input = (data_path+'/CCLE/graph_data/' + 'radius' + str(radius) + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    n_fingerprint = len(fingerprint_dict)

    """Create a dataset and split it into train/dev/test."""
    graph_dataset = list(zip(compounds, adjacencies))

    """Load CCLE data."""
    rma, var, smiles = untils.load_CCLE_data(data_path)

    smiles_vals = smiles["smiles"].values
    smiles_index = smiles.index
    cell_names = rma.index.values
    gene = rma.columns.values
    drugs_num = len(smiles_index)

    GDSC_drug_dict = untils.get_drug_dict(smiles)

    """Load CCLE drug similarity data."""

    drug_similarity = pd.read_csv(data_path+"/CCLE/drug_similarity/CCLE_drug_similarity.csv", header=None)
    similarity_softmax = torch.from_numpy(drug_similarity.to_numpy().astype(np.float32))
    similarity_softmax = similarity_softmax.to(device)

    """split dataset"""
    data = pd.read_csv(data_path+"/CCLE/CCLE_Data/CCLE_cell_drug_labels.csv", index_col=0)
    ccle_smiles = pd.read_csv(data_path+"/CCLE/CCLE_Data/CCLE_smiles.csv", index_col=0)

    if gParameters['data_type'] == 'original':
        train_id, test_id = untils.split_data(data,split_case=split_case, ratio=0.9,cell_names=cell_names) # gihan
    elif gParameters['data_type'] == 'ccle_candle':
        if gParameters['data_split_seed'] > -1:
            all_id = pd.read_csv(data_path+'/CCLE/CCLE_Data/CCLE_cell_drug_labels.csv')
            all_id = all_id.sort_values(by='cell_line_id')
            all_id.reset_index(drop=True, inplace=True)
            train_id, val_id = train_test_split(all_id, test_size=0.2, random_state=gParameters['data_split_seed'])
            test_id, val_id = train_test_split(val_id, test_size=0.5, random_state=gParameters['data_split_seed'])
            train_id.reset_index(drop=True, inplace=True)
            test_id.reset_index(drop=True, inplace=True)
            val_id.reset_index(drop=True, inplace=True)
        else:
            train_id = pd.read_csv(data_path+'/CCLE/CCLE_Data/train.csv')
            val_id = pd.read_csv(data_path+'/CCLE/CCLE_Data/val.csv')
            test_id = pd.read_csv(data_path+'/CCLE/CCLE_Data/test.csv')
            

    dataset_sizes = {'train': len(train_id), 'test': len(test_id)}
    print(dataset_sizes['train'], dataset_sizes['test'])
    log.info('train size = {:d},test size = {:d}\n'.format(dataset_sizes['train'], dataset_sizes['test']))
    # log.flush()

    trainDataset = CreateDataset(rma, var, train_id)
    valDataset = CreateDataset(rma, var, val_id)
    testDataset = CreateDataset(rma, var, test_id)

    # Dataloader
    train_loader = Data.DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE * len(device_ids), shuffle=True)
    val_loader = Data.DataLoader(dataset=valDataset, batch_size=BATCH_SIZE * len(device_ids), shuffle=False , drop_last=False)
    test_loader = Data.DataLoader(dataset=testDataset, batch_size=BATCH_SIZE * len(device_ids), shuffle=False, drop_last=False)

    """create SWnet model"""
    model_ft = Model(dim, layer_gnn, drugs_num, n_fingerprint, similarity_softmax,
            GDSC_drug_dict, graph_dataset, n_genes)  # gihan
    log.info(str(model_ft))

    """cuda"""
    model_ft = model_ft.cuda(device=device_ids[0])  #

    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=LR)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()

    """start training model !"""
    print("start training model")

    model_ft, best_weights = train_model(model_ft, train_loader, val_loader, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                           output_dir, file_name, num_epochs=num_epochs)

    print("loading best weights....")
    model_ft.load_state_dict(best_weights)
    print("loading best weights done.")
    
    st_test = time.time()
    test_mse, test_r2, df_res_test = eval_model(model_ft, test_loader, ccle_smiles)
    test_time = time.time() - st_test

    val_mse, val_r2, _ = eval_model(model_ft, val_loader, ccle_smiles)

    print('test rmse:{},r2:{}'.format(test_mse**.5, test_r2))
    print('val rmse:{},r2:{}'.format(val_mse**.5, val_r2))
    # log.info('rmse:{},r2:{}'.format(mse**.5, r2))
    print("preprocess time: ", pp_time)
    print("inference time: ", test_time)
    
    test_scores = {"val_loss": val_mse, "r2": val_r2 }
    with open( os.path.join(output_dir,"test_scores.json"), "w", encoding="utf-8") as f:
        json.dump(test_scores, f, ensure_ascii=False, indent=4)
    # add_natoms(df_res)
    df_res_test.to_csv(os.path.join(output_dir,"test_predictions.csv"), index=False)

    """Save the gene weights """
    fuse = pd.DataFrame(model_ft.fuse_weight.cpu().detach().numpy(),
                        index=smiles_index, columns=gene)

    os.makedirs(os.path.join(output_dir, "log/logs/gene_weights/"), exist_ok=True)
    fuse_name = os.path.join(output_dir, 'log/logs/gene_weights/' + str(round(test_mse, 4)) + '_' + file_name + '_r' + str(radius) + '_s' + str(split_case) + '.csv')
    fuse.to_csv(fuse_name)
    print("Save the gene weights done!")

    return test_scores


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
    handler = logging.FileHandler( os.path.join(CANDLE_DATA_DIR, gParameters['model_name'], 'Output', 'log.txt'), mode='w' )
    fomatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
    handler.setFormatter(fomatter)
    log.addHandler(handler)

    scores = run(gParameters)
    print("Done.")

