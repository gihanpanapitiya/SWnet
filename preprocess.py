import pandas as pd
import improve_utils
from sklearn.preprocessing import StandardScaler

def get_drug_response_data(df, metric):
    
    # df = rs_train.copy()
    smiles_df = improve_utils.load_smiles_data()
    data_smiles_df = pd.merge(df, smiles_df, on = "improve_chem_id", how='left') 
    data_smiles_df = data_smiles_df.dropna(subset=[metric])
    data_smiles_df = data_smiles_df[['improve_sample_id', 'smiles', 'improve_chem_id', metric]]
    data_smiles_df = data_smiles_df.drop_duplicates()
    data_smiles_df = data_smiles_df.reset_index(drop=True)

    return data_smiles_df


def sava_split_files(df, file_name):

    tmp = df[['improve_sample_id', 'improve_chem_id', 'ic50']]
    tmp = tmp.rename(columns={'improve_sample_id':'cell_line_id',
                        'improve_chem_id':'drug_id',
                        'ic50':'labels'})
    tmp.to_csv(file_name, index=False)

def candle_preprocess(data_type='CCLE', 
                         metric='ic50', 
                         data_path=None,
                         ext_gene_file=None

):
        
    # data_type='CCLE'
    # metric='ic50'
    rs_all = improve_utils.load_single_drug_response_data(source=data_type, split=0,
                                                        split_type=["train", "test", 'val'],
                                                        y_col_name=metric)

    rs_train = improve_utils.load_single_drug_response_data(source=data_type,
                                                            split=0, split_type=["train"],
                                                            y_col_name=metric)
    rs_test = improve_utils.load_single_drug_response_data(source=data_type,
                                                        split=0,
                                                        split_type=["test"],
                                                        y_col_name=metric)
    rs_val = improve_utils.load_single_drug_response_data(source=data_type,
                                                        split=0,
                                                        split_type=["val"],
                                                        y_col_name=metric)



    train_df = get_drug_response_data(rs_train, metric)
    val_df = get_drug_response_data(rs_val, metric)
    test_df = get_drug_response_data(rs_test, metric)

    
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    all_df.reset_index(drop=True, inplace=True)

    sava_split_files(train_df, data_path+'/CCLE/CCLE_Data/train.csv')
    sava_split_files(val_df, data_path+'/CCLE/CCLE_Data/val.csv')
    sava_split_files(test_df, data_path+'/CCLE/CCLE_Data/test.csv')



    smi_candle = all_df[['improve_chem_id', 'smiles']]
    smi_candle.drop_duplicates(inplace=True)
    smi_candle.reset_index(drop=True, inplace=True)
    smi_candle.set_index('improve_chem_id', inplace=True)
    smi_candle.index.name=None
    smi_candle.to_csv(data_path+'/CCLE/CCLE_Data/CCLE_smiles.csv')

    mutation_data = improve_utils.load_cell_mutation_data(gene_system_identifier="Entrez")
    expr_data = improve_utils.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
    mutation_data = mutation_data.reset_index()
    gene_data = mutation_data.columns[1:]

    ext_genes = pd.read_csv(ext_gene_file,index_col=0)
    common_genes=list(set(gene_data).intersection(ext_genes.columns))

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
    expr.to_csv(data_path+'/CCLE/CCLE_Data/CCLE_RNAseq.csv')

    mut.index.name=None
    mut.to_csv(data_path+'/CCLE/CCLE_Data/CCLE_DepMap.csv')

    all_df=all_df[['improve_sample_id', 'improve_chem_id', 'ic50']]
    all_df=all_df.rename(columns={'improve_sample_id':'cell_line_id',
                        'improve_chem_id':'drug_id',
                        'ic50':'labels'})
    all_df.to_csv(data_path+'/CCLE/CCLE_Data/CCLE_cell_drug_labels.csv', index=False)