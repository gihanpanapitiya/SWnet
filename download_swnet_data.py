import os
import urllib
import candle
from data_utils import Downloader, candle_data_dict


CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")

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
     }
]

required = None


def get_data(data_url, cache_subdir, download=True, svn=False):
    print('downloading data')
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


# def download_csa_data(opt):

#     csa_data_folder = os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data', 'csa_data', 'raw_data')
#     splits_dir = os.path.join(csa_data_folder, 'splits') 
#     x_data_dir = os.path.join(csa_data_folder, 'x_data')
#     y_data_dir = os.path.join(csa_data_folder, 'y_data')

#     if not os.path.exists(csa_data_folder):
#         print('creating folder: %s'%csa_data_folder)
#         os.makedirs(csa_data_folder)
#         os.mkdir( splits_dir  )
#         os.mkdir( x_data_dir  )
#         os.mkdir( y_data_dir  )
    

#     for file in ['CCLE_all.txt', 'CCLE_split_0_test.txt', 'CCLE_split_0_train.txt', 'CCLE_split_0_val.txt']:
#         urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/splits/{file}',
#         splits_dir+f'/{file}')

#     for file in ['cancer_mutation_count.txt', 'drug_SMILES.txt','drug_ecfp4_512bit.txt', 'cancer_gene_expression.txt']:
#         urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/x_data/{file}',
#         x_data_dir+f'/{file}')

#     for file in ['response.txt']:
#         urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/y_data/{file}',
#         y_data_dir+f'/{file}')


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

    data_url = gParameters['data_url']
    data_version = gParameters['data_version']
    data_type = candle_data_dict[gParameters['data_source']]
    data_split_id = gParameters['data_split_id']

    data_path=os.path.join(CANDLE_DATA_DIR, gParameters['model_name'], 'Data')

    get_data(data_url, os.path.join(data_path, 'swn_original'), True, False)

    downloader = Downloader(data_version)
    downloader.download_candle_data(data_type=data_type, split_id=data_split_id, data_dest=data_path)
    # download_csa_data(gParameters)
    print("Done.")

