# Candle Compatible SWnet
<!-- : a deep learning model for drug response prediction from cancer genomic signatures and compound chemical structures -->

The Candle compatible code for the paper "SWnet: a deep learning model for drug response prediction from cancer genomic signatures and compound chemical structures" by Zhaorui Zuo, Penglei Wang, Xiaowei Chen, Li Tian, Hui Ge & Dahong Qian.

## Install Instructions
### Using Singularity
To build the Singularity container, run
```
singularity build --fakeroot SWnet.sif SWnet.def,
```
where SWnet.sif is the name of the Singularity container and SWnet.def is the Singularity definition file provided in this repository

### Using Conda
```
conda env create -f environment.yaml
```


## Running the model using the original author's data
Set the CANDLE_DATA_DIR and CUDA_VISIBLE_DEVICES environment variables.

1. Download and process data
Make sure data_source is set to 'ccle_original' in the swnet_ccle_model.txt. Yet to test the model with original GDSC data.

```
python preprocess.py
```
```
singularity exec --nv SWnet.sif preprocess.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt
```

2. Train the model:
```
python SWnet_CCLE_baseline_pytorch.py
```
```
singularity exec --nv SWnet.sif train.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt
```

3. Get predictions:
```
python infer.py
```
```
singularity exec --nv SWnet.sif infer.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt
```




## Running the model for CSA models for within-study validation
Set the CANDLE_DATA_DIR and CUDA_VISIBLE_DEVICES environment variables.

1. Download and process data. 
Set the following parameters in the swnet_ccle_model.txt
```
data_source =  ccle_candle # choose one from these : 'ccle_candle', 'gcsi_candle', 'gdscv1_candle', 'gdscv2_candle', 'ctrpv2_candle'
cross_study=False
data_split_id=0
metric='auc' or 'ic50'
```
Then run the following command,
```
python preprocess.py
```
```
singularity exec --nv SWnet.sif preprocess.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt
```


2. Train the model:
```
python SWnet_CCLE_baseline_pytorch.py
```
```
singularity exec --nv SWnet.sif train.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt
```


3. Get predictions:
```
python infer.py
```
```
singularity exec --nv SWnet.sif infer.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt
```


## Running the model for CSA models for cross-study validation

1. Download and process data. 
Set the following parameters in the swnet_ccle_model.txt. We have to set cross_study=True for this case.
For the 'data_source' field, provide the name of the dataset which the model is trained on.
```
data_source = ctrpv2_candle # the name of the data_set the model will be trained with (chose one from these: 'ccle_candle' 'gcsi_candle', 'gdscv1_candle', 'gdscv2_candle', 'ctrpv2_candle')
```
In the 'other_ds' field, provide the datasets which the trained model will be evaluated with.
```
other_ds = 'ccle_candle, gcsi_candle, gdscv1_candle, gdscv2_candle' # other datasets the trained model will be tested with. specify these datasets seperated by a comma. eg: 'ccle_candle' 'gcsi_candle', 'gdscv1_candle'
```
```
cross_study=True
data_split_id=0
metric='auc' # or 'ic50'
```

Then run the following command,
```
python preprocess.py
```

```
singularity exec --nv SWnet.sif preprocess.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt
```
This will take a while.

2. Train the model:
```
python SWnet_CCLE_baseline_pytorch.py
```
```
singularity exec --nv SWnet.sif train.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt
```


3. Get predictions:
Change the data_source to what you want to test on and run infer.py as follows.
```
python infer.py --data_source ccle_candle
```
```
singularity exec --nv SWnet.sif infer.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt --data_source name-of-the-evaluating-dataset
```
For example, 
```
singularity exec --nv SWnet.sif infer.sh  $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR --config_file /path/to/*_model.txt --data_source ccle_candle
```



