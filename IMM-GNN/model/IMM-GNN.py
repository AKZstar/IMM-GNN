import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from model.act_func import act_dict
from model.layer_utils import *


def PreGNN(dim_in, dim_out, num_layers):
    """
    Wrapper for NN layer before GNN message passing

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    """
    # dim_inner = int(cfg.dim_inner_cof * dim_out)
    # num_layers = cfg.preGNN_num_layers
    return GeneralMultiLinearLayer(num_layers,
                                   dim_in,
                                   dim_out,
                                   dim_inner=dim_out,
                                   final_act=True,
                                   need_layernorm=False)

class GNN_mol_atom_aggregate(nn.Module):
    def __init__(self, atom_feature_dim, need_Linear_trans=True):
        super().__init__()
        self.atom_feature_dim = atom_feature_dim
        self.need_Linear_trans = need_Linear_trans
        self.attn_type = cfg.global_attn_type
        self.dropout = nn.Dropout(cfg.dropout)
        if need_Linear_trans:
            self.mol_atom_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
            self.mol_atom_neighbor_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)

        self.GNN_attn_act = act_dict[cfg.GNN_attn_act]
        self.need_layer_norm = cfg.layer_norm
        self.layer_norm = nn.LayerNorm(self.atom_feature_dim)

        self.head_nums = cfg.gloabal_head_nums

        self.d_k = self.atom_feature_dim // self.head_nums

        if self.attn_type == 'GAT':
            self.mol_GAT_align = nn.Linear(2 * self.d_k, 1)
        elif self.attn_type == 'GATV2':
            self.mol_GATV2_align = nn.Linear(self.d_k, 1)

    def forward(self, mol_atom_feature, atom_feature, mol_atom_attend_mask, mol_atom_softmax_mask):
        # print(self.head_nums)
        # print(self.d_k)

        if self.need_Linear_trans == True:
            mol_atom_feature = self.dropout(mol_atom_feature)
            atom_feature = self.dropout(atom_feature)

            mol_atom_feature = self.mol_atom_fc(mol_atom_feature)
            atom_feature = self.mol_atom_neighbor_fc(atom_feature)

        batch_size, mol_atom_length, atom_feature_dim = atom_feature.shape
        mol_atom_expand = mol_atom_feature.unsqueeze(-2).expand(batch_size, mol_atom_length, atom_feature_dim)

        mol_atom_expand = mol_atom_expand.view(batch_size, mol_atom_length, self.head_nums, self.d_k)
        atom_feature = atom_feature.view(batch_size, mol_atom_length, self.head_nums, self.d_k)

        mol_atom_softmax_mask = mol_atom_softmax_mask.unsqueeze(-2).expand(batch_size, mol_atom_length, self.head_nums, 1)
        mol_atom_attend_mask = mol_atom_attend_mask.unsqueeze(-2).expand(batch_size, mol_atom_length, self.head_nums, 1)

        if self.attn_type == 'GAT':
            print('GAT')
            mol_feature_align = torch.cat([mol_atom_expand, atom_feature], dim=-1)
            mol_align_score = self.GNN_attn_act(self.mol_GAT_align(mol_feature_align))
            mol_align_score = mol_align_score + mol_atom_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -3)
            ############只加了这一句，用完记得注释掉，尝试不用softmax
            # mol_attention_weight = mol_align_score

            mol_attention_weight = mol_attention_weight * mol_atom_attend_mask
            mol_atom_context = torch.sum(torch.mul(mol_attention_weight, atom_feature), -3)

            mol_atom_context = mol_atom_context.contiguous().view(batch_size, self.head_nums * self.d_k)
        elif self.attn_type == 'GATV2':
            # print('GATV2')
            mol_feature_align = mol_atom_expand + atom_feature
            mol_feature_align = self.GNN_attn_act(mol_feature_align)
            mol_align_score = self.mol_GATV2_align(mol_feature_align)
            mol_align_score = mol_align_score + mol_atom_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -3)
            mol_attention_weight = mol_attention_weight * mol_atom_attend_mask
            mol_atom_context = torch.sum(torch.mul(mol_attention_weight, atom_feature), -3)

            mol_atom_context = mol_atom_context.contiguous().view(batch_size, self.head_nums * self.d_k)
        elif self.attn_type == 'dot_trans':
            print('dot_trans')
            mol_align_score = self.GNN_attn_act((mol_atom_expand * atom_feature).sum(dim=-1))

            mol_align_score = mol_align_score / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

            mol_align_score = mol_align_score.unsqueeze(-1)

            mol_align_score = mol_align_score + mol_atom_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -3)
            mol_attention_weight = mol_attention_weight * mol_atom_attend_mask
            mol_atom_context = torch.sum(torch.mul(mol_attention_weight, atom_feature), -3)

            mol_atom_context = mol_atom_context.contiguous().view(batch_size, self.head_nums * self.d_k)

        if self.need_layer_norm == True:
            mol_atom_context = self.layer_norm(mol_atom_context)

        return mol_atom_context


class GNN_atom_aggregate(nn.Module):
    def __init__(self, atom_feature_dim, atom_neighbor_feature_dim, new_atom_fea_dim):
        super().__init__()
        self.atom_feature_dim = atom_feature_dim  ##原子特征维度
        self.atom_neighbor_feature_dim = atom_neighbor_feature_dim  ##原子的邻居特征维度

        self.new_atom_fea_dim = new_atom_fea_dim  ##原子的邻居特征维度

        self.dropout = nn.Dropout(cfg.dropout)  ##指定dropout数值
        self.atom_fc = nn.Linear(self.atom_feature_dim, self.new_atom_fea_dim)  ##对原子特征进行线性映射，维度大小不变
        self.atom_neighbor_fc = nn.Linear(self.atom_neighbor_feature_dim, self.new_atom_fea_dim)  ##将原来的原子的邻居特征映射到与原子特征维度大小一样

        self.attn_type = cfg.atom_attn_type

        self.GNN_attn_act = act_dict[cfg.GNN_attn_act]

        self.layer_norm = nn.LayerNorm(self.atom_feature_dim)
        self.need_layer_norm = cfg.layer_norm

        self.head_nums = cfg.atom_head_nums

        self.d_k = self.new_atom_fea_dim // self.head_nums

        if self.attn_type == 'GAT':
            self.atom_GAT_align = nn.Linear(2 * self.d_k, 1)
        elif self.attn_type == 'GATV2':
            self.atom_GATV2_align = nn.Linear(self.d_k, 1)

    def forward(self, atom_feature, atom_neighbor_feature, atom_softmax_mask, atom_attend_mask):
        # print(self.head_nums)
        # print(self.d_k)
        atom_feature = self.dropout(atom_feature)
        atom_neighbor_feature = self.dropout(atom_neighbor_feature)

        atom_feature = self.atom_fc(atom_feature)
        atom_neighbor_feature = self.atom_neighbor_fc(atom_neighbor_feature)

        batch_size, mol_atom_num, max_atom_neighbor_num, atom_fingerprint_dim = atom_neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                                atom_fingerprint_dim)
        # atom_context = None

        if self.attn_type == 'GAT':
            print('GAT')
            atom_feature_expand = atom_feature_expand.view(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                           self.head_nums, self.d_k)
            atom_neighbor_feature = atom_neighbor_feature.view(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                               self.head_nums, self.d_k)

            atom_feature_align = torch.cat([atom_feature_expand, atom_neighbor_feature], dim=-1)
            atom_align_score = self.GNN_attn_act(self.atom_GAT_align(atom_feature_align))

            atom_softmax_mask = atom_softmax_mask.unsqueeze(-2).expand(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                                       self.head_nums, 1)
            atom_attend_mask = atom_attend_mask.unsqueeze(-2).expand(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                                     self.head_nums, 1)

            atom_align_score = atom_align_score + atom_softmax_mask
            # atom_align_score.softmax()
            atom_attention_weight = F.softmax(atom_align_score, -3)
            atom_attention_weight = atom_attention_weight * atom_attend_mask
            atom_context = torch.sum(torch.mul(atom_attention_weight, atom_neighbor_feature), -3)

            atom_context = atom_context.contiguous().view(batch_size, mol_atom_num, self.head_nums * self.d_k)
        elif self.attn_type == 'GATV2':
            # print('GATV2')
            atom_feature_expand = atom_feature_expand.view(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                           self.head_nums, self.d_k)
            atom_neighbor_feature = atom_neighbor_feature.view(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                               self.head_nums, self.d_k)

            feat_align = atom_feature_expand + atom_neighbor_feature
            feat_align = self.GNN_attn_act(feat_align)
            atom_align_score = self.atom_GATV2_align(feat_align)

            atom_softmax_mask = atom_softmax_mask.unsqueeze(-2).expand(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                                       self.head_nums, 1)
            atom_attend_mask = atom_attend_mask.unsqueeze(-2).expand(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                                     self.head_nums, 1)

            atom_align_score = atom_align_score + atom_softmax_mask
            atom_attention_weight = F.softmax(atom_align_score, -3)
            atom_attention_weight = atom_attention_weight * atom_attend_mask
            atom_context = torch.sum(torch.mul(atom_attention_weight, atom_neighbor_feature), -3)

            atom_context = atom_context.contiguous().view(batch_size, mol_atom_num, self.head_nums * self.d_k)

        elif self.attn_type == 'dot_trans':
            print('dot_trans')
            atom_feature_expand = atom_feature_expand.view(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                           self.head_nums, self.d_k)
            atom_neighbor_feature = atom_neighbor_feature.view(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                               self.head_nums, self.d_k)

            atom_align_score = self.GNN_attn_act((atom_feature_expand * atom_neighbor_feature).sum(dim=-1))

            atom_align_score = atom_align_score / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

            atom_align_score = atom_align_score.unsqueeze(-1)

            # print('d_k: ', self.d_k)

            atom_softmax_mask = atom_softmax_mask.unsqueeze(-2).expand(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                                       self.head_nums, 1)
            atom_attend_mask = atom_attend_mask.unsqueeze(-2).expand(batch_size, mol_atom_num, max_atom_neighbor_num,
                                                                     self.head_nums, 1)

            atom_align_score = atom_align_score + atom_softmax_mask
            # atom_align_score.softmax()
            atom_attention_weight = F.softmax(atom_align_score, -3)
            atom_attention_weight = atom_attention_weight * atom_attend_mask
            atom_context = torch.sum(torch.mul(atom_attention_weight, atom_neighbor_feature), -3)

            atom_context = atom_context.contiguous().view(batch_size, mol_atom_num, self.head_nums * self.d_k)

            # atom_context = self.update_fc(atom_context)
            #
            # atom_context = self.GNN_update_context_act(atom_context)

        if self.need_layer_norm == True:
            atom_context = self.layer_norm(atom_context)

        return atom_context


class GNN_mol_atom_update(nn.Module):
    def __init__(self, atom_feature_dim):
        super().__init__()
        self.update_type = cfg.global_update_type
        self.atom_feature_dim = atom_feature_dim
        self.GNN_update_act = act_dict[cfg.GNN_update_act]
        if self.update_type == 'skipconcat':
            self.skipconcat_trans = nn.Linear(2 * self.atom_feature_dim, self.atom_feature_dim)
        elif self.update_type == 'GRU':
            self.atom_GRUCell = nn.GRUCell(self.atom_feature_dim, self.atom_feature_dim)

    def forward(self, mol_atom_feature, mol_atom_context):
        # mol_activated_atom_feature = None  # 初始化为None
        # mol_atom_state = None
        if self.update_type == 'skipsum':
            mol_atom_state = mol_atom_feature + mol_atom_context
            mol_activated_atom_feature = self.GNN_update_act(mol_atom_state)
        elif self.update_type == 'skipconcat':
            mol_atom_state = torch.cat([mol_atom_feature, mol_atom_context], dim=-1)
            mol_activated_atom_feature = self.GNN_update_act(self.skipconcat_trans(mol_atom_state))
        elif self.update_type == 'GRU':
            mol_atom_state = self.atom_GRUCell(mol_atom_context, mol_atom_feature)
            mol_activated_atom_feature = self.GNN_update_act(mol_atom_state)

        return  mol_activated_atom_feature


class MinGRU(nn.Module):
    def __init__(self, input_dim_x, input_dim_h):
        super(MinGRU, self).__init__()

        self.linear_z = nn.Linear(input_dim_x, input_dim_h)

        self.linear_h = nn.Linear(input_dim_x, input_dim_h)

    def forward(self, x_t, h_prev):
        # x_t: (batch_size, input_size)
        # h_prev: (batch_size, hidden_size)

        z_t = torch.sigmoid(self.linear_z(x_t))
        h_tilde = self.linear_h(x_t)

        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        return h_t

class GNN_atom_update(nn.Module):
    def __init__(self, atom_feature_dim):
        super().__init__()
        self.update_type = cfg.atom_update_type
        self.atom_feature_dim = atom_feature_dim
        self.GNN_update_act = act_dict[cfg.GNN_update_act]
        if self.update_type == 'skipconcat':
            self.skipconcat_trans = nn.Linear(2 * self.atom_feature_dim, self.atom_feature_dim)
        elif self.update_type == 'GRU':
            self.atom_GRUCell = nn.GRUCell(self.atom_feature_dim, self.atom_feature_dim)
        elif self.update_type == 'minGRU':
            self.minGRU = MinGRU(self.atom_feature_dim, self.atom_feature_dim)
        elif self.update_type == 'actGRU':
            self.atom_GRUCell = nn.GRUCell(self.atom_feature_dim, self.atom_feature_dim)
        elif self.update_type == 'actminGRU':
            self.minGRU = MinGRU(self.atom_feature_dim, self.atom_feature_dim)
        elif self.update_type == 'act_linear_skipsum':
            self.skipsum_trans = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)

    def forward(self, atom_feature, atom_context):
        if self.update_type == 'skipsum':
            atom_state = atom_feature + atom_context
            activated_atom_feature = self.GNN_update_act(atom_state)
        elif self.update_type == 'act_linear_skipsum':
            atom_state = atom_feature + self.skipsum_trans(self.GNN_update_act(atom_context))
            activated_atom_feature = self.GNN_update_act(atom_state)
        elif self.update_type == 'act_skipsum':
            atom_state = atom_feature + self.GNN_update_act(atom_context)
            activated_atom_feature = self.GNN_update_act(atom_state)
        elif self.update_type == 'skipconcat':
            atom_state = torch.cat([atom_feature, atom_context], dim=-1)
            activated_atom_feature = self.GNN_update_act(self.skipconcat_trans(atom_state))
        elif self.update_type == 'GRU':
            batch_size, mol_atom_num, atom_feature_dim = atom_feature.shape
            atom_context_reshape = atom_context.view(batch_size * mol_atom_num, atom_feature_dim)
            atom_feature_reshape = atom_feature.view(batch_size * mol_atom_num, atom_feature_dim)
            atom_feature_reshape = self.atom_GRUCell(atom_context_reshape, atom_feature_reshape)
            atom_state = atom_feature_reshape.view(batch_size, mol_atom_num, atom_feature_dim)
            activated_atom_feature = self.GNN_update_act(atom_state)
        elif self.update_type == 'minGRU':
            batch_size, mol_atom_num, atom_feature_dim = atom_feature.shape

            atom_state = self.minGRU(atom_context, atom_feature)
            activated_atom_feature = self.GNN_update_act(atom_state)
        elif self.update_type == 'actGRU':
            batch_size, mol_atom_num, atom_feature_dim = atom_feature.shape
            atom_context_reshape = atom_context.view(batch_size * mol_atom_num, atom_feature_dim)
            atom_feature_reshape = atom_feature.view(batch_size * mol_atom_num, atom_feature_dim)

            atom_context_reshape = self.GNN_update_act(atom_context_reshape)

            atom_feature_reshape = self.atom_GRUCell(atom_context_reshape, atom_feature_reshape)
            atom_state = atom_feature_reshape.view(batch_size, mol_atom_num, atom_feature_dim)
            activated_atom_feature = self.GNN_update_act(atom_state)
        elif self.update_type == 'actminGRU':
            batch_size, mol_atom_num, atom_feature_dim = atom_feature.shape

            atom_context = self.GNN_update_act(atom_context)

            atom_state = self.minGRU(atom_context, atom_feature)
            activated_atom_feature = self.GNN_update_act(atom_state)

        return activated_atom_feature

class Layer_attention_by_mol(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_feature_dim = cfg.atom_feature_dim
        self.mol_atom_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
        self.mol_atom_neighbor_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
        self.mol_layer_GAT_align = nn.Linear(2 * self.atom_feature_dim, 1)

        self.GNN_attn_act = act_dict[cfg.GNN_attn_act]

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, mol_layer_feature_by_atom_sum, layer_neighbor_feature):
        mol_layer_feature_by_atom_sum = self.dropout(mol_layer_feature_by_atom_sum)
        layer_neighbor_feature = self.dropout(layer_neighbor_feature)
        mol_layer_feature_by_atom_sum = self.mol_atom_fc(mol_layer_feature_by_atom_sum)
        layer_neighbor_feature = self.mol_atom_neighbor_fc(layer_neighbor_feature)


        batch_size, layer_nums, mol_atom_fingerprint_dim = layer_neighbor_feature.shape
        mol_layer_feature_by_atom_expand = mol_layer_feature_by_atom_sum.unsqueeze(-2).expand(batch_size, layer_nums,
                                                                                          mol_atom_fingerprint_dim)

        mol_layer_feature_align = torch.cat([mol_layer_feature_by_atom_expand, layer_neighbor_feature], dim=-1)
        mol_layer_align_score = self.GNN_attn_act(self.mol_layer_GAT_align(mol_layer_feature_align))
        mol_layer_attention_weight = F.softmax(mol_layer_align_score, -2)  ##因为layer中的每一层都是有值得，没有闲置和多余得
        mol_layer_atom_context = torch.sum(torch.mul(mol_layer_attention_weight, layer_neighbor_feature), -2)

        return mol_layer_atom_context

class Layer_attention_by_mol_channel(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_feature_dim = cfg.atom_feature_dim
        self.mol_atom_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
        self.mol_atom_neighbor_fc = nn.Linear(self.atom_feature_dim, self.atom_feature_dim)
        self.mol_layer_GAT_align = nn.Linear(2 * self.atom_feature_dim, self.atom_feature_dim)

        self.GNN_attn_act = act_dict[cfg.GNN_attn_act]

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, mol_layer_feature_by_atom_sum, layer_neighbor_feature):
        mol_layer_feature_by_atom_sum = self.dropout(mol_layer_feature_by_atom_sum)
        layer_neighbor_feature = self.dropout(layer_neighbor_feature)
        mol_layer_feature_by_atom_sum = self.mol_atom_fc(mol_layer_feature_by_atom_sum)
        layer_neighbor_feature = self.mol_atom_neighbor_fc(layer_neighbor_feature)


        batch_size, layer_nums, mol_atom_fingerprint_dim = layer_neighbor_feature.shape
        mol_layer_feature_by_atom_expand = mol_layer_feature_by_atom_sum.unsqueeze(-2).expand(batch_size, layer_nums,
                                                                                          mol_atom_fingerprint_dim)

        mol_layer_feature_align = torch.cat([mol_layer_feature_by_atom_expand, layer_neighbor_feature], dim=-1)
        mol_layer_align_score = self.GNN_attn_act(self.mol_layer_GAT_align(mol_layer_feature_align))
        mol_layer_attention_weight = F.softmax(mol_layer_align_score, -2)  ##因为layer中的每一层都是有值得，没有闲置和多余得
        mol_layer_atom_context = torch.sum(torch.mul(mol_layer_attention_weight, layer_neighbor_feature), -2)

        return mol_layer_atom_context




class ChannelAttention(nn.Module):
    def __init__(self, fea_dim, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 全局最大池化

        # MLP：通过全连接层生成注意力权重
        self.fc = nn.Sequential(
            nn.Linear(fea_dim, fea_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fea_dim // reduction, fea_dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 激活函数

    def forward(self, x):
        # 输入 x 的形状为 (batchsize, mol_atom_num, fea_dim)
        batchsize, mol_atom_num, fea_dim = x.size()

        # 对 mol_atom_num 维度进行池化
        x_perm = x.transpose(1, 2)  # (batchsize, fea_dim, mol_atom_num)
        avg_out = self.avg_pool(x_perm).squeeze(-1)  # 全局平均池化 (batchsize, fea_dim)
        max_out = self.max_pool(x_perm).squeeze(-1)  # 全局最大池化 (batchsize, fea_dim)

        # 通过全连接层生成通道注意力权重
        avg_out = self.fc(avg_out)  # (batchsize, fea_dim)
        max_out = self.fc(max_out)  # (batchsize, fea_dim)

        # 最终的通道注意力权重
        scale = self.sigmoid(avg_out + max_out)  # 结合平均池化和最大池化的结果

        # 对通道进行重新校准：按通道加权原始特征
        scale = scale.unsqueeze(1).expand_as(x)  # 调整形状以便与输入相乘 (batchsize, 1, fea_dim) -> (batchsize, mol_atom_num, fea_dim)
        out = x * scale  # 利用广播机制逐元素相乘，重新校准原子特征
        return out


class GNN_atom_bond(nn.Module):
    def __init__(self, output_units_num):
        super().__init__()

        self.atom_feature_dim = cfg.atom_feature_dim
        self.bond_feature_dim = cfg.bond_feature_dim

        self._0hop_fea_dim = (self.atom_feature_dim // 8) * (4- cfg.hop_coff)
        self._1hop_fea_dim = (self.atom_feature_dim // 2)
        self._2hop_fea_dim = (self.atom_feature_dim // 8) * cfg.hop_coff

        self.radius = cfg.radius
        self.atom_neighbor_feature_dim = self.atom_feature_dim

        self.atom_preGNN = PreGNN(cfg.atom_dim_in, self.atom_feature_dim, cfg.preGNN_num_layers)
        self.bond_preGNN = PreGNN(cfg.bond_dim_in, self.bond_feature_dim, cfg.preGNN_num_layers)

        self.get_1hop_atom_context_first = GNN_atom_aggregate(self.atom_feature_dim, self.atom_feature_dim + self.bond_feature_dim, self._1hop_fea_dim)
        self.get_2hop_atom_context_first = GNN_atom_aggregate(self.atom_feature_dim, self.atom_feature_dim + self.bond_feature_dim, self._2hop_fea_dim)

        self.get_1hop_atom_context = nn.ModuleList(
            [GNN_atom_aggregate(self.atom_feature_dim, self.atom_neighbor_feature_dim, self._1hop_fea_dim) for r in range(self.radius-1)])
        self.get_2hop_atom_context = nn.ModuleList(
            [GNN_atom_aggregate(self.atom_feature_dim, self.atom_neighbor_feature_dim, self._2hop_fea_dim) for r in range(self.radius - 1)])

        self.get_0hop_atom_context = nn.ModuleList(
            [nn.Linear(self.atom_feature_dim, self._0hop_fea_dim) for r in range(self.radius)])

        self.channelAttention = nn.ModuleList(
            [ChannelAttention(self.atom_feature_dim, cfg.channel_reduction) for r in range(self.radius)])

        self.get_atom_update = GNN_atom_update(self.atom_feature_dim)

        self.get_mol_atom_context = nn.ModuleList([GNN_mol_atom_aggregate(self.atom_feature_dim) for r in range(self.radius)])

        self.get_mol_atom_update = nn.ModuleList([GNN_mol_atom_update(cfg.atom_feature_dim) for r in range(self.radius)])

        self.dropout = nn.Dropout(p=cfg.dropout)

        self.postGNN_classifier = nn.Linear(self.atom_feature_dim, output_units_num)
        
        self.layer_atten = Layer_attention_by_mol_channel()
        
        self.layer_update_act = act_dict[cfg.GNN_update_act]

    def forward(self, x_atoms, x_bonds, atom_1hop_neighbors_atom_index, atom_1hop_neighbors_bond_index, atom_2hop_neighbors_atom_index, atom_2hop_neighbors_bond_index,atom_mask):
        global mol_feature_by_atom, mol_feature_by_bond

        batch_size, mol_atom_num, len_atom_feat = x_atoms.size()

        atom_feature = self.atom_preGNN(x_atoms)  ##原子特征先经过一系列线性映射层，类似于embedding层
        bond_feature = self.bond_preGNN(x_bonds)

        atom_1hop_attend_mask, atom_1hop_softmax_mask = get_atom_attend_and_softmax_mask(atom_1hop_neighbors_atom_index, mol_atom_num)
        atom_2hop_attend_mask, atom_2hop_softmax_mask = get_atom_attend_and_softmax_mask(atom_2hop_neighbors_atom_index, mol_atom_num)


        mol_atom_attend_mask, mol_atom_softmax_mask = get_mol_atom_attend_and_softmax_mask(atom_mask)
        
        # mol_fea = 0
        
        mol_layer_fea = []
        layer_atom_feature = []

        for i in range(self.radius):
            if i==0:
                atom_1hop_neighbor_feature = get_atom_neighbor_feature_atom_bond(atom_feature, bond_feature,
                                                                            atom_1hop_neighbors_atom_index,
                                                                            atom_1hop_neighbors_bond_index)
                atom_2hop_neighbor_feature = get_atom_neighbor_feature_atom_bond(atom_feature, bond_feature,
                                                                            atom_2hop_neighbors_atom_index,
                                                                            atom_2hop_neighbors_bond_index)

                atom_1hop_context = self.get_1hop_atom_context_first(atom_feature, atom_1hop_neighbor_feature, atom_1hop_softmax_mask,
                                                        atom_1hop_attend_mask)
                atom_2hop_context = self.get_2hop_atom_context_first(atom_feature, atom_2hop_neighbor_feature,
                                                                     atom_2hop_softmax_mask,
                                                                     atom_2hop_attend_mask)
            else:

                atom_1hop_neighbor_feature = get_atom_neighbor_feature_atom(atom_feature, atom_1hop_neighbors_atom_index)
                atom_2hop_neighbor_feature = get_atom_neighbor_feature_atom(atom_feature, atom_2hop_neighbors_atom_index)


                atom_1hop_context = self.get_1hop_atom_context[i-1](atom_feature, atom_1hop_neighbor_feature,
                                                                     atom_1hop_softmax_mask,
                                                                     atom_1hop_attend_mask)
                atom_2hop_context = self.get_2hop_atom_context[i-1](atom_feature, atom_2hop_neighbor_feature,
                                                                     atom_2hop_softmax_mask,
                                                                     atom_2hop_attend_mask)

            atom_0hop_context = self.get_0hop_atom_context[i](atom_feature)

            atom_context = torch.concat([atom_0hop_context, atom_1hop_context, atom_2hop_context], dim=-1)
            # atom_context = self.channelAttention[i](atom_context) + atom_context

            atom_feature = self.get_atom_update(atom_feature, atom_context)

            # if i==0:##计算初始的基于原子和化学键的分子表示，这里选择的是直接求和，再经过了一个若干层的embedding层；
            mol_feature_by_atom = torch.sum(atom_feature * mol_atom_attend_mask, dim=-2)

            mol_context_by_atom = self.get_mol_atom_context[i](mol_feature_by_atom, atom_feature, mol_atom_attend_mask,
                                                                mol_atom_softmax_mask)
            mol_feature_by_atom = self.get_mol_atom_update[i](mol_feature_by_atom, mol_context_by_atom)

            mol_layer_fea.append(mol_feature_by_atom)

        
        layer_neighbor_feature = torch.stack(mol_layer_fea, dim=1)

        if cfg.layer_atten_query == 'all_layer_sum':  # 'final_layer'
            mol_layer_feature_by_atom_sum = torch.sum(layer_neighbor_feature, dim=-2)
        elif cfg.layer_atten_query == 'final_layer':
            mol_layer_feature_by_atom_sum = mol_layer_fea[cfg.radius - 1]
        
        mol_layer_context = self.layer_atten(mol_layer_feature_by_atom_sum, layer_neighbor_feature)
        mol_final_fea = self.layer_update_act(mol_layer_context + mol_layer_feature_by_atom_sum)


        output = self.postGNN_classifier(self.dropout(mol_final_fea))

        return atom_feature, output




