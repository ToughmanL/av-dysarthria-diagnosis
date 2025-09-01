import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.model.ops.basic_ops import ModelFusion
from networks.model.audio_visual.attention_layer import attbase, attnormbase, justatt, withoutatt

from collections import OrderedDict
import os


def get_attlayer(atttype):
  if atttype == 'withoutatt':
    return withoutatt
  elif atttype == 'justatt':
    return justatt
  elif atttype == 'base':
    return attbase
  elif atttype == 'attnormbase':
    return attnormbase

class CrossAVFusion(nn.Module):
  def __init__(self, 
               fold: int,
               audio_net: torch.nn.Module, 
               video_net: torch.nn.Module,
               audio_dim: int,
               video_dim: int,
               output_size: int,
               fusion_type: 'crossatt',
               crosstype: 'AV',
               atttype: 'base',
               similiarity: 'scaled_dot',
               dropout: 0.1):
    super().__init__()
    self.audio_net = audio_net
    self.visual_net = video_net
    self.crosstype = crosstype
    self.fusion_net = ModelFusion(fusion_type, audio_dim, dim=1)
    if fusion_type == 'concat':
      av_dim = audio_dim + video_dim
    else:
      av_dim = audio_dim
    
    self.relu = nn.ReLU()
    self.a_bn_1d = nn.BatchNorm1d(audio_dim)
    self.v_bn_1d = nn.BatchNorm1d(video_dim)
    attentionLayer = get_attlayer(atttype)
    self.corssAV = attentionLayer(d_model=audio_dim, nhead=2, dropout=0.03, similiarity=similiarity)
    self.corssVA = attentionLayer(d_model=video_dim, nhead=2, dropout=0.03, similiarity=similiarity)
    self.selfAV = attentionLayer(d_model=av_dim, nhead=2, dropout=0.03, similiarity=similiarity)
    self.head = nn.Linear(av_dim, output_size)
  
  def forward(self, 
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor):
    a = self.audio_net(speech, speech_lens)
    v = self.visual_net(speech, video, speech_lens)
    av = self.crossav(a, v)
    selfav = self.selfAV(av, av)
    out = self.head(selfav)
    return out

  def crossav(self, a, v):
    if self.crosstype == 'AV':
      acv = self.corssAV(a, v)
      vca = self.corssVA(v, a)
      av = self.fusion_net(acv, vca)
      return av
    elif self.crosstype == 'Av':
      acv = self.corssAV(a, v)
      return acv
    elif self.crosstype == 'aV':
      vca = self.corssVA(v, a)
      return vca

class CatAVFusion(nn.Module):
  def __init__(self, 
               fold: int,
               audio_net: torch.nn.Module, 
               video_net: torch.nn.Module,
               audio_dim: int,
               video_dim: int,
               fusion_type: str,
               output_size: int):
    super().__init__()
    self.audio_net = audio_net
    self.visual_net = video_net
    self.fusion_net = ModelFusion(fusion_type, audio_dim, dim=1)
    if fusion_type == 'concat':
      av_dim = audio_dim + video_dim
    else:
      av_dim = audio_dim
    
    self.relu = nn.ReLU()
    self.a_bn_1d = nn.BatchNorm1d(audio_dim)
    self.v_bn_1d = nn.BatchNorm1d(video_dim)
    self.relu = nn.ReLU()
    self.head = nn.Linear(av_dim, output_size)
  
  def forward(self, 
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor):
    a = self.audio_net(speech, speech_lens)
    v = self.visual_net(speech, video, speech_lens)
    # a = self.a_bn_1d(self.relu(a))
    # v = self.v_bn_1d(self.relu(v))
    av = torch.cat((a, v), dim=1)
    out = self.head(av)
    return out


class ShareAVFusion(nn.Module):
  def __init__(self, 
              fold: int,
              audio_net: torch.nn.Module, 
              video_net: torch.nn.Module,
              audio_dim: int,
              video_dim: int,
              output_size: int,
              fusion_type: 'crossatt',
              crosstype: 'AV',
              atttype: 'base',
              similiarity: 'scaled_dot',
              dropout: 0.1):
    super().__init__()
    self.audio_net = audio_net
    self.visual_net = video_net
    self.fusion_net = ModelFusion(fusion_type, audio_dim, dim=1)
    if fusion_type == 'concat':
      av_dim = audio_dim + video_dim
    else:
      av_dim = audio_dim
    self.dropout = nn.Dropout(0.3)
    self.norm1 = nn.LayerNorm(audio_dim)
    self.norm2 = nn.LayerNorm(video_dim)
    self.head = nn.Linear(av_dim, output_size)

  def forward(self, 
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor):
    a = self.audio_net(speech, speech_lens)
    v = self.visual_net(speech, video, speech_lens)
    a_norm = self.dropout(self.norm1(a))
    v_norm = self.dropout(self.norm2(v))
    a_space = a + torch.mul(a, v_norm)
    v_space = v + torch.mul(v, a_norm)
    av = torch.cat((a_space, v_space), dim=1)
    out = self.head(av)
    return out


class ShareCrossAVFusion(nn.Module):
  def __init__(self, 
              fold: int,
              audio_net: torch.nn.Module, 
              video_net: torch.nn.Module,
              audio_dim: int,
              video_dim: int,
              output_size: int,
              fusion_type: 'crossatt',
              crosstype: 'AV',
              atttype: 'base',
              similiarity: 'scaled_dot',
              dropout):
    super().__init__()
    self.audio_net = audio_net
    self.visual_net = video_net
    self.crosstype = crosstype
    self.fusion_net = ModelFusion(fusion_type, audio_dim, dim=1)
    if fusion_type == 'concat':
      av_dim = audio_dim + video_dim
    else:
      av_dim = audio_dim
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(audio_dim)
    self.norm2 = nn.LayerNorm(video_dim)
    attentionLayer = get_attlayer(atttype)
    self.corssAV = attentionLayer(d_model=audio_dim, nhead=2, dropout=0.03, similiarity=similiarity)
    self.corssVA = attentionLayer(d_model=video_dim, nhead=2, dropout=0.03, similiarity=similiarity)
    self.selfAV = attentionLayer(d_model=av_dim, nhead=2, dropout=0.03, similiarity=similiarity)
    self.head = nn.Linear(av_dim, output_size)

  def forward(self, 
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor):
    a = self.audio_net(speech, speech_lens)
    v = self.visual_net(speech, video, speech_lens)
    a_norm = self.dropout(self.norm1(a))
    v_norm = self.dropout(self.norm2(v))
    a_space = a + torch.mul(a, v_norm)
    v_space = v + torch.mul(v, a_norm)
    av = self.crossav(a_space, v_space)
    selfav = self.selfAV(av, av)
    out = self.head(selfav)
    return out

  def crossav(self, a, v):
    if self.crosstype == 'AV':
      acv = self.corssAV(a, v)
      vca = self.corssVA(v, a)
      av = self.fusion_net(acv, vca)
      return av
    elif self.crosstype == 'Av':
      acv = self.corssAV(a, v)
      return acv
    elif self.crosstype == 'aV':
      vca = self.corssVA(v, a)
      return vca


if __name__ == '__main__':
  import yaml
  from networks.utils.init_model import init_model

  fold = 0
  config_file = 'conf/shareAV_resnetseq10_tsnsegsplit.yaml'
  with open(config_file, 'r') as fin:
      configs = yaml.load(fin, Loader=yaml.FullLoader)
  model, configs = init_model(configs, fold)

  speech = torch.randn(64, 200, 80)
  speech_len = torch.tensor([64, 200])
  video_data = torch.randn([64, 61, 3, 96, 96])
  # text_data = torch.zeros([64, 114])
  # text_len = torch.tensor([64, 2])
  label = torch.tensor([64])

  out = model(speech, video_data, speech_len, label)
  print(out.shape)