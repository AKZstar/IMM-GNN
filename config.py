import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class Config():

    ##############输入输出文件路径#####################

    hyper_log_file = "./hyper_log_file/BBBP_scaffoldsplit"
    task_name = 'BBBP'
    tasks = ['BBBP']
    data_raw_filename = "./data/BBBP.csv"
    seed_number = 667
    batch_size = 64
    epochs = 150
    learning_rate = 5e-4
    lr_FACTOR = 0.8
    lr_PATIENCE = 15
    lr_MIN_LR = 1e-5
    early_roc_epochs = 50
    early_loss_epochs = 30
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr_scheduler_type = 'reduce'
    atom_feature_dim = 160
    bond_feature_dim = 64
    hop_coff = 2 
    radius = 2  
    preGNN_num_layers = 1
    channel_reduction = 2
    GNN_update_act = 'lrelu'
    GNN_attn_act = 'lrelu'
    atom_attn_type = 'GATV2'
    atom_update_type = 'actminGRU'
    atom_head_nums = 1
    global_attn_type = 'GATV2'
    global_update_type = 'skipsum'
    gloabal_head_nums = 8
    layer_atten_query = 'final_layer'
    preGNN_act = 'lrelu'
    dropout = 0.3
    layer_norm = False
    weight_decay = 1e-5
    atom_dim_in = 39 
    bond_dim_in = 10 
    per_task_output_units_num = 2
    max_evals = 40
    hyper_rank_num = 10

cfg = Config()



