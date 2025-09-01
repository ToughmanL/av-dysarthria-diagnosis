import torch
import torch.nn as nn
from networks.model.audio_resnet.resnet import get_resnet
from networks.model.audio_resnet.resnetsqu import get_resnetfus, get_resnetsqu
from networks.model.audio_resnet.res2net import get_res2net
from networks.model.ops.basic_ops import ConsensusModule


class ResSeg2(nn.Module):
    def __init__(self, out_dim, base_model='resnet', resnum=10, feat_dim=80, consensus_type='avg', dropout=0.5):
        super(ResSeg2, self).__init__()
        self.base_model = base_model
        self.consensus_type = consensus_type
        self.dropout = dropout
        self.out_dim = out_dim

        self.consensus = ConsensusModule(consensus_type)
        if base_model == 'resnet':
            self.base_model = get_resnet(resnum=resnum, feat_dim=feat_dim, embed_dim=out_dim)
        elif base_model == 'res2net':
            self.base_model = get_resnet(resnum=resnum, feat_dim=feat_dim, embed_dim=out_dim)
        
        self.part1 = self.base_model
        self.part2 = self.base_model

    
    def forward(self, x, x_len):
        part1_data, part2_data = x.split(x.size(1) // 2, dim=1)
        part1_out = self.part1(part1_data, x_len)
        part2_out = self.part2(part2_data, x_len)
        out = torch.stack([part1_out, part2_out], dim=1)
        out = self.consensus(out)
        return out.squeeze()
    
    def output_size(self):
        return self.out_dim


class ResSeg3(nn.Module):
    def __init__(self, out_dim, base_model='resnet', resnum=10, feat_dim=80, consensus_type='avg', dropout=0.5):
        super(ResSeg3, self).__init__()
        self.base_model = base_model
        self.consensus_type = consensus_type
        self.dropout = dropout
        self.out_dim = out_dim

        self.consensus = ConsensusModule(consensus_type)
        if base_model == 'resnet':
            self.base_model = get_resnet(resnum=resnum, feat_dim=feat_dim, embed_dim=out_dim)
        elif base_model == 'res2net':
            self.base_model = get_resnet(resnum=resnum, feat_dim=feat_dim, embed_dim=out_dim)
        
        self.part1 = self.base_model
        self.part2 = self.base_model
        self.part3 = self.base_model

    
    def forward(self, x, x_len):
        part1_data, part2_data, part3_data = x.split(x.size(1) // 3, dim=1)
        part1_out = self.part1(part1_data, x_len)
        part2_out = self.part2(part2_data, x_len)
        part3_out = self.part3(part3_data, x_len)
        out = torch.stack([part1_out, part2_out, part3_out], dim=1)
        out = self.consensus(out)
        return out.squeeze()
    
    def output_size(self):
        return self.out_dim


class ResSeg4(nn.Module):
    def __init__(self, out_dim, base_model='resnet', resnum=10, feat_dim=80, consensus_type='avg', dropout=0.5):
        super(ResSeg4, self).__init__()
        self.base_model = base_model
        self.consensus_type = consensus_type
        self.dropout = dropout
        self.out_dim = out_dim

        self.consensus = ConsensusModule(consensus_type)
        if base_model == 'resnet':
            self.base_model = get_resnet(resnum=resnum, feat_dim=feat_dim, embed_dim=out_dim)
        elif base_model == 'res2net':
            self.base_model = get_resnet(resnum=resnum, feat_dim=feat_dim, embed_dim=out_dim)
        
        self.part1 = self.base_model
        self.part2 = self.base_model
        self.part3 = self.base_model
        self.part4 = self.base_model

    
    def forward(self, x, x_len):
        part1_data, part2_data, part3_data, part4_data = x.split(x.size(1) // 4, dim=1)
        part1_out = self.part1(part1_data, x_len)
        part2_out = self.part2(part2_data, x_len)
        part3_out = self.part3(part3_data, x_len)
        part4_out = self.part4(part4_data, x_len)
        out = torch.stack([part1_out, part2_out, part3_out, part4_out], dim=1)
        out = self.consensus(out)
        return out.squeeze()
    
    def output_size(self):
        return self.out_dim


class ResSeg5(nn.Module):
    def __init__(self, out_dim, base_model='resnet', resnum=10, feat_dim=80, consensus_type='avg', dropout=0.5):
        super(ResSeg5, self).__init__()
        self.base_model = base_model
        self.consensus_type = consensus_type
        self.dropout = dropout
        self.out_dim = out_dim

        self.consensus = ConsensusModule(consensus_type)
        if base_model == 'resnet':
            self.base_model = get_resnet(resnum=resnum, feat_dim=feat_dim, embed_dim=out_dim)
        elif base_model == 'res2net':
            self.base_model = get_resnet(resnum=resnum, feat_dim=feat_dim, embed_dim=out_dim)
        
        self.part1 = self.base_model
        self.part2 = self.base_model
        self.part3 = self.base_model
        self.part4 = self.base_model
        self.part5 = self.base_model

    
    def forward(self, x, x_len):
        part1_data, part2_data, part3_data, part4_data, part5_data = x.split(x.size(1) // 5, dim=1)
        part1_out = self.part1(part1_data, x_len)
        part2_out = self.part2(part2_data, x_len)
        part3_out = self.part3(part3_data, x_len)
        part4_out = self.part4(part3_data, x_len)
        part5_out = self.part5(part3_data, x_len)
        out = torch.stack([part1_out, part2_out, part3_out, part4_out, part5_out], dim=1)
        out = self.consensus(out)
        return out.squeeze()
    
    def output_size(self):
        return self.out_dim


def get_resnet10seg(segnum, out_dim, base_model='resnet', resnum=10, feat_dim=80, consensus_type='avg', dropout=0.5):
    if segnum == 2:
        return ResSeg2(out_dim, base_model, resnum, feat_dim, consensus_type, dropout)
    elif segnum == 3:
        return ResSeg3(out_dim, base_model, resnum, feat_dim, consensus_type, dropout)
    elif segnum == 4:
        return ResSeg4(out_dim, base_model, resnum, feat_dim, consensus_type, dropout)
    elif segnum == 5:
        return ResSeg5(out_dim, base_model, resnum, feat_dim, consensus_type, dropout)