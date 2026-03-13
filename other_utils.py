import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem
from sklearn import metrics
from sklearn.model_selection import train_test_split
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.optim.lr_scheduler import _LRScheduler
from config import cfg
import logging
from random import Random
import pandas as pd

def seed_everything(seed=2021):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_label_balanced_weight(tasks,remained_df):
    weights = []
    for i, task in enumerate(tasks):
        negative_df = remained_df[remained_df[task] == 0][["smiles", task]]
        positive_df = remained_df[remained_df[task] == 1][["smiles", task]]
        weights.append([(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0], \
                        (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]])

    return weights

def random_split(remained_df, random_seed=0):
    test_df = remained_df.sample(frac=1 / 10, random_state=random_seed)  # test set
    training_data = remained_df.drop(test_df.index)  # training data

    # training data is further divided into validation set and train set
    valid_df = training_data.sample(frac=1 / 9, random_state=random_seed)  # validation set
    train_df = training_data.drop(valid_df.index)  # train set
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, valid_df, test_df


def scaffold_split(df, frac=None, balanced=False, include_chirality=False, ramdom_state=cfg.seed_number):
    frac = [0.8, 0.1, 0.1] if frac is None else frac
    assert sum(frac) == 1

    n_total_valid = int(np.floor(frac[1] * len(df)))
    n_total_test = int(np.floor(frac[2] * len(df)))
    n_total_train = len(df) - n_total_valid - n_total_test

    scaffolds_sets = defaultdict(list)
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["cano_smiles"])
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaffolds_sets[scaffold].append(idx)

    # Put stuff that's bigger than half the val/test size into train, rest just order randomly
    random = Random(ramdom_state)
    if balanced:
        index_sets = list(scaffolds_sets.values())
        big_index_sets, small_index_sets = list(), list()
        for index_set in index_sets:
            if len(index_set) > n_total_valid / 2 or len(index_set) > n_total_test / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(ramdom_state)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffolds_sets.values()), key=lambda index_set: len(index_set), reverse=True)

    train_index, valid_index, test_index = list(), list(), list()
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= n_total_train:
            train_index += index_set
        elif len(valid_index) + len(index_set) <= n_total_valid:
            valid_index += index_set
        else:
            test_index += index_set

    # print(train_index, valid_index, test_index)

    train_df = df.iloc[train_index].reset_index(drop=True)
    valid_df = df.iloc[valid_index].reset_index(drop=True)
    test_df = df.iloc[test_index].reset_index(drop=True)

    return train_df, valid_df, test_df


def create_logger(name: str, save_dir: str = None, log_filename: str = "default.log", quiet: bool = False) -> logging.Logger:
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # 设置为INFO级别，之前是DEBUG级别
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 设置为INFO级别
    logger.addHandler(ch)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fh = logging.FileHandler(os.path.join(save_dir, log_filename))
        fh.setLevel(logging.INFO)  # 设置为INFO级别

        logger.addHandler(fh)

    return logger

def print_model_pm(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.

    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


def random_atom_bond_traindf(df):
    # 2. 创建两份副本并添加前缀
    df_atom = df.copy()
    df_atom['cano_smiles'] = 'atom_' + df_atom['cano_smiles']

    df_bond = df.copy()
    df_bond['cano_smiles'] = 'bond_' + df_bond['cano_smiles']

    # 3. 合并原始DataFrame和修改后的DataFrame
    combined_df = pd.concat([df_atom, df_bond], ignore_index=True)

    # 4. 随机打乱行顺序
    shuffled_df = combined_df.sample(frac=1,random_state=cfg.seed_number).reset_index(drop=True)
    return shuffled_df


def rbf_transform(x, centers, gamma):
    """
    Radial Basis Function transformation

    Args:
        x (float): Input scalar value
        centers (list or np.ndarray): RBF centers, shape (n_centers,)
        gamma (float): RBF scale parameter

    Returns:
        transformed_features (np.ndarray): Transformed features, shape (n_centers,)
    """
    x = np.array([[x]])  # Shape (1, 1)
    # print(x)
    centers = np.array(centers).reshape(1, -1)  # Shape (1, n_centers)
    return np.exp(-gamma * np.square(x - centers)).reshape(-1)  # Shape (n_centers,)

class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, init_lr, max_lr, final_lr):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]