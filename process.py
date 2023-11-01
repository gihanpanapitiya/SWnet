import pandas as pd
# import improve_utils
import data_utils
import os
import urllib
from sklearn.preprocessing import StandardScaler
from data_utils import remove_smiles_with_noneighbor_frags
from data_utils import DataProcessor, add_smiles, load_generic_expression_data

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



# def add_smiles(data_dir, df, metric):
    
#     # df = rs_train.copy()
#     smiles_df = data_utils.load_smiles_data(data_dir)
#     data_smiles_df = pd.merge(df, smiles_df, on = "improve_chem_id", how='left') 
#     data_smiles_df = data_smiles_df.dropna(subset=[metric])
#     data_smiles_df = data_smiles_df[['improve_sample_id', 'smiles', 'improve_chem_id', metric]]
#     data_smiles_df = data_smiles_df.drop_duplicates()
#     data_smiles_df.dropna(inplace=True)
#     data_smiles_df = data_smiles_df.reset_index(drop=True)

#     return data_smiles_df


def save_split_files(df, file_name, metric='ic50'):

    tmp = df[['improve_sample_id', 'improve_chem_id', metric]]
    tmp = tmp.rename(columns={'improve_sample_id':'cell_line_id',
                        'improve_chem_id':'drug_id',
                        metric:'labels'})
    tmp.to_csv(file_name, index=False)

def candle_preprocess(data_type='CCLE', 
                         metric='ic50',
                         split_id=0, 
                         data_path=None,
                         ext_gene_file=None, params=None):
        
    # data_type='CCLE'
    # metric='ic50'
#     rs_all = improve_utils.load_single_drug_response_data(source=data_type, split=0,
#                                                         split_type=["train", "test", 'val'],
#                                                         y_col_name=metric)

    # rs_train = improve_utils.load_single_drug_response_data(source=data_type,
    #                                                         split=0, split_type=["train"],
    #                                                         y_col_name=metric)
    # rs_test = improve_utils.load_single_drug_response_data(source=data_type,
    #                                                     split=0,
    #                                                     split_type=["test"],
    #                                                     y_col_name=metric)
    # rs_val = improve_utils.load_single_drug_response_data(source=data_type,
    #                                                     split=0,
    #                                                     split_type=["val"],
    #                                                     y_col_name=metric)

    data_processor = DataProcessor(params['data_version'])


    rs_train = data_processor.load_drug_response_data(data_path, data_type=data_type, split_id=split_id, split_type='train', response_type=metric)
    rs_val = data_processor.load_drug_response_data(data_path, data_type=data_type, split_id=split_id, split_type='val', response_type=metric)
    rs_test = data_processor.load_drug_response_data(data_path, data_type=data_type, split_id=split_id, split_type='test', response_type=metric)


    print("data shape: ", rs_train.shape, rs_val.shape, rs_test.shape)
    # rs_train = data_utils.load_drug_response_data(data_path, data_type=data_type, split_id=split_id, split_type='train', 
    #         response_type=metric, sep="\t", dropna=True)
    # rs_val = data_utils.load_drug_response_data(data_path, data_type=data_type, split_id=split_id, split_type='val', 
    #         response_type=metric, sep="\t", dropna=True)
    # rs_test = data_utils.load_drug_response_data(data_path, data_type=data_type, split_id=split_id, split_type='test', 
    #         response_type=metric, sep="\t", dropna=True)



    smiles_df = data_processor.load_smiles_data(data_path)
    smiles_df = remove_smiles_with_noneighbor_frags(smiles_df)
    # smiles_path = os.path.join(data_path, 'drug_SMILES.tsv')
    # smiles_df.to_csv(smiles_path, sep='\t', index=False)

    train_df = add_smiles(smiles_df,rs_train, metric)
    val_df = add_smiles(smiles_df,rs_val, metric)
    test_df = add_smiles(smiles_df,rs_test, metric)

    if params['use_proteomics_data']:

        gexp_ = load_generic_expression_data('proteomics_restructure_with_knn_impute.tsv')
        use_improve_ids = gexp_.index.values

        train_df = train_df[train_df.improve_sample_id.isin(use_improve_ids)]
        val_df = val_df[val_df.improve_sample_id.isin(use_improve_ids)]
        test_df = test_df[test_df.improve_sample_id.isin(use_improve_ids)]



    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    all_df.reset_index(drop=True, inplace=True)

    save_split_files(train_df, data_path+f'/{data_type}/{data_type}_Data/train.csv', metric)
    save_split_files(val_df, data_path+f'/{data_type}/{data_type}_Data/val.csv', metric)
    save_split_files(test_df, data_path+f'/{data_type}/{data_type}_Data/test.csv', metric)



    smi_candle = all_df[['improve_chem_id', 'smiles']]
    # smi_candle = data_utils.load_smiles_data(data_path)
    smi_candle.drop_duplicates(inplace=True)
    smi_candle.reset_index(drop=True, inplace=True)
    smi_candle.set_index('improve_chem_id', inplace=True)
    smi_candle.index.name=None
    smi_candle.to_csv(data_path+f'/{data_type}/{data_type}_Data/{data_type}_smiles.csv')

    smi_candle = data_processor.load_smiles_data(data_path)
    smi_candle.drop_duplicates(inplace=True)
    smi_candle.reset_index(drop=True, inplace=True)
    smi_candle.set_index('improve_chem_id', inplace=True)
    smi_candle.index.name=None
    smi_candle.to_csv(data_path+f'/{data_type}/{data_type}_Data/all_smiles.csv')




    mutation_data = data_processor.load_cell_mutation_data(data_dir=data_path, gene_system_identifier="Entrez")
    expr_data = data_processor.load_gene_expression_data(data_dir=data_path, gene_system_identifier="Gene_Symbol")
    mutation_data = mutation_data.reset_index()
    
    if not params['use_proteomics_data']:
        print('using gene expression data')
        gene_data = mutation_data.columns[1:]
    else:
        print('using proteomics data')
        expr_data = load_generic_expression_data('proteomics_restructure_with_knn_impute.tsv')
        gene_data = expr_data.columns
        gene_data = list(set(gene_data).intersection(mutation_data.columns[1:]))
        print("gene data: ", len(gene_data))

    ext_genes = pd.read_csv(ext_gene_file,index_col=0)
    common_genes=sorted( list(set(gene_data).intersection(ext_genes.columns)) )
    # common_genes=list(set(gene_data).intersection(ext_genes))

    mut = mutation_data[mutation_data.improve_sample_id.isin(all_df.improve_sample_id)]
    

    mut = mut.loc[:, ['improve_sample_id'] + common_genes ]
    mut.improve_sample_id.nunique() == mut.shape[0]
    mut.set_index('improve_sample_id', inplace=True)
    mut = mut.gt(0).astype(int)

    expr = expr_data[expr_data.index.isin(mut.index)]
    expr = expr.loc[:, common_genes]

    sc = StandardScaler()

    expr[:] = sc.fit_transform(expr[:])


    expr.index.name=None
    expr.to_csv(data_path+f'/{data_type}/{data_type}_Data/{data_type}_RNAseq.csv')

    mut.index.name=None
    mut.to_csv(data_path+f'/{data_type}/{data_type}_Data/{data_type}_DepMap.csv')

    all_df=all_df[['improve_sample_id', 'improve_chem_id', metric]]
    all_df=all_df.rename(columns={'improve_sample_id':'cell_line_id',
                        'improve_chem_id':'drug_id',
                        metric:'labels'})
    all_df.to_csv(data_path+f'/{data_type}/{data_type}_Data/{data_type}_cell_drug_labels.csv', index=False)
