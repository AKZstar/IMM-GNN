# IMM-GNN

Source code for **IMM-GNN**: An Integrative Multi-hop and Multi-scale Graph Neural Network for Molecular Property Prediction.

## Requirements

The environment configuration required to run this project can be found in the `environment.yml` file.

## Files

### data
We use five benchmark datasets for molecular property prediction, including four classification datasets (BACE, BBBP, ClinTox, and SIDER) and two regression datasets (ESOL and Lipophilicity).

All datasets are obtained from [MoleculeNet](https://moleculenet.org/).

### preprocess

**Featurizer.py**
Extracts initial atom and bond features based on RDKit.

**getFeatures_molecule.py**
Models molecules as graph-structured data and extracts atom and bond feature matrices, 1-hop and 2-hop neighbor index matrices, and atom mask arrays.

**preprocess_prog.ipynb**
A data preprocessing notebook that performs SMILES canonicalization, deduplication, and invalid molecule filtering.
It generates two pickle files:
- `BBBP.pickle` (feature dictionary)
- `BBBP_remained_df.pickle` (cleaned dataset)

### model

**act_func.py**
An activation function registry for flexible selection across different model layers.

**layer_utils.py**
A utility library for reusable low-level neural network components.

**IMM-GNN.py**
Provides a complete implementation of the proposed IMM-GNN model.

### Root-level files

**other_utils.py**
Provides auxiliary utilities for training, including scaffold-based data splitting, class-balanced weight calculation, and random seed control.

**config.py**
Implements a `Config` class for centralized management of all project parameters, including model hyperparameters and training settings.

**run_main.py**
The main entry point of the project, responsible for the complete workflow of model training, validation, and testing.

## Quick Start

### 1. Data Preprocessing

Run the following notebook:

```bash
jupyter notebook preprocess/preprocess_prog.ipynb
```

Execute all cells to generate:
- `data/BBBP.pickle` (feature dictionary)
- `data/BBBP_remained_df.pickle` (cleaned dataset)

### 2. Model Training

Run the training script:

```bash
python run_main.py
```

> **Note:** To switch datasets, modify the following fields in `config.py`:
> ```python
> task_name = 'BACE'                    # Dataset name
> tasks = ['Class']                     # Task label column(s)
> data_raw_filename = "./data/BACE.csv" # Data file path
> ```
