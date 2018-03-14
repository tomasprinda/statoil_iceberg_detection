# Statoil Iceberg Detection
Solution for Kaggle competition [Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/statoil)

# Overview
 - Tensorflow

# Install
Project is organized as a python package. So first you should install it best with deveploment mode

```bash
python setup.py develop
```

Data are stored in separate folder so set the variable `BASE_DIR` in [statoil/cfg.py](statoil/cfg.py#L3) to your data dir.

# Project structure
- [scripts/](scripts/) contains app entrypoints
  - [scripts/prepare_data.py](scripts/prepare_data.py) Runs the preprocessing pipeline 
  - [scripts/train.py](scripts/train.py) Train a model. Takes a configuration file which contains a a model class and hyperparametrs to use with the model.
  - [scripts/hypertrain.py](scripts/hypertrain.py) Special version of a a `train.py` with a possibility to seach hyperparametrs.
  - [scripts/predict.py](scripts/predict.py) Uses a trained model to predict a testset to submit to Kaggle.
- [statoil/models/](statoil/models/) Contains definition of models
- [conf](conf/) Contains configuration files
- [notebooks/](notebooks)
  - [notebooks/Data examine.ipynb](notebooks/Data examine.ipynb) To see the data before doing anything else
  - [notebooks/Eval.ipynb](notebooks/Eval.ipynb) Notebook for evaluation, see errors, confusion matrix, etc.
  
- `BASE_DIR` Dir with data
  - `BASE_DIR/raw/` Constains raw data downloaded from Kaggle
  - `BASE_DIR/DATAxx/` Preprocessed data where xx is a version of data 
  - `BASE_DIR/models/` Dir that contains downloaded models from tensorflow/slim library used for transfer learning.
  - `BASE_DIR/experiments/` Experiment results
  


# Preprocessing
[Download data](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data) and unzip in `BASEDIR/raw/` (without subfolders).

Train data split to `train` and `dev` datasets. These datasets also expanded by flipping and rotation.

```bash
python scripts/prepare_data.py
``` 

# Run experiment
Train
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --conf conf/simple_cnn.json --exp simple_cnn --long_run
```

Predict
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py --exp simple_cnn
```

# Best results
[Best results](conf/conf_best.json) achieved log_loss on dev_set = 0.1963, on test_set = 0.1946.  

# What I tried
 - Training on traindev set before submit: :(
 - Batch normalization :(
 - Pretrain network on a sythetic data :(
 - Transfer learning :(
 - Dense net :(
 - Hyperparameter tuning :)
 



