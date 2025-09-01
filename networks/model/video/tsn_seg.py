# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
from torch import nn
from networks.model.video.tsn import TSN
from networks.model.ops.basic_ops import ConsensusModule
from networks.model.ops.transforms import *


# random select index
def select_index(data, num_segments, rgb_len=1):
  rgb_data_list, temporal_data_list = [], []
  for i in range(num_segments):
    if isinstance(data, list):
      seg_data = data[i]
    elif isinstance(data, torch.Tensor):
      seg_data = data[:, i, ...]
    start, end = int(seg_data.size(1) * 0.1), int(seg_data.size(1) * 0.9)
    # select_data = seg_data[:, int((start+end)/2)]
    select_data = seg_data[:, 3]
    rgb_data_list.append(select_data.unsqueeze(1))
    temporal_data_list.append(seg_data)
  return (rgb_data_list, temporal_data_list)

# segment data and two stream
def segment_data(input, segments_len):
  segment_num = len(segments_len)
  if segment_num == 4: # 包括构音障碍划分和随机划分
    # 将数据分为四段，第一段为静音段10帧，第二段为AIT段15帧，第四段为后段7帧，中间段为剩余帧
    front_len, ait_len, middle_len, back_len = segments_len
    front_data = input[:, :front_len]
    ait_data = input[:, front_len:front_len + ait_len]
    middle_data = input[:, front_len + ait_len:-back_len]
    back_data = input[:, -back_len:]
    input_data = [front_data, ait_data, middle_data, back_data]
    return select_index(input_data, segment_num)
  else: # 随机划分
    data_list = []
    start = 0
    for i in range(segment_num):
      data_list.append(input[:, start:start+segments_len[i]])
      start += segments_len[i]
    return select_index(data_list, segment_num)


class TSNSEG(nn.Module):
  def __init__(self, num_class, out_dim, num_segments, modality='RGB',
         time_base_model='resnet50', space_base_model='resnet50',
         new_length=61, consensus_type='avg', dropout=0.5):
    super(TSNSEG, self).__init__()
    self.modality = modality.split("+")
    if len(self.modality) > 1:
      self.modality_type = 'twostream'
      self.space_input_type = self.modality[0]
      self.time_input_type = self.modality[1]
    elif self.modality[0] == 'RGB':
      self.modality_type = 'space'
      self.space_input_type = self.modality[0]
    else:
      self.modality_type = 'time'
      self.time_input_type = self.modality[0]
    
    if num_segments == -4: # 按照构音障碍来分段
      self.segments_len = [10, 15, 29, 7]
    else: # 按照num_segments来分段
      base_len = new_length // num_segments
      add_len = new_length % num_segments
      self.segments_len = [base_len + 1 if i < add_len else base_len for i in range(num_segments)]
    self.consensus = ConsensusModule(consensus_type)

    self.space_branch, self.time_branch = [], []
    if len(self.segments_len) == 4:
      if self.modality_type == 'twostream' or self.modality_type == 'space':
        self.tns_space_1 = TSN(out_dim, 1, self.space_input_type, space_base_model, 1, dropout=dropout)
        self.tns_space_2 = TSN(out_dim, 1, self.space_input_type, space_base_model, 1, dropout=dropout)
        self.tns_space_3 = TSN(out_dim, 1, self.space_input_type, space_base_model, 1, dropout=dropout)
        self.tns_space_4 = TSN(out_dim, 1, self.space_input_type, space_base_model, 1, dropout=dropout)
      if self.modality_type == 'twostream' or self.modality_type == 'time':
        self.tsn_time_1 = TSN(out_dim, 1, self.time_input_type, time_base_model, self.segments_len[0], dropout=dropout)
        self.tsn_time_2 = TSN(out_dim, 1, self.time_input_type, time_base_model, self.segments_len[1], dropout=dropout)
        self.tsn_time_3 = TSN(out_dim, 1, self.time_input_type, time_base_model, self.segments_len[2], dropout=dropout)
        self.tsn_time_4 = TSN(out_dim, 1, self.time_input_type, time_base_model, self.segments_len[3], dropout=dropout)

    self.fc = nn.Linear(out_dim, num_class)
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU(inplace=True)
    
  def forward(
        self,
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lengths: torch.Tensor
    ): # 4 segments
    input = video
    rgb_data, flow_data = segment_data(input, self.segments_len)
    rgb_out_list, flow_out_list = [], []
    
    if self.modality_type == 'twostream' or self.modality_type == 'space':
      rgb_out_list.append(self.tns_space_1(speech, rgb_data[0], speech_lengths).unsqueeze(1))
      rgb_out_list.append(self.tns_space_2(speech, rgb_data[1], speech_lengths).unsqueeze(1))
      rgb_out_list.append(self.tns_space_3(speech, rgb_data[2], speech_lengths).unsqueeze(1))
      rgb_out_list.append(self.tns_space_4(speech, rgb_data[3], speech_lengths).unsqueeze(1))
    if self.modality_type == 'twostream' or self.modality_type == 'time':
      flow_out_list.append(self.tsn_time_1(speech, flow_data[0], speech_lengths).unsqueeze(1))
      flow_out_list.append(self.tsn_time_2(speech, flow_data[1], speech_lengths).unsqueeze(1))
      flow_out_list.append(self.tsn_time_3(speech, flow_data[2], speech_lengths).unsqueeze(1))
      flow_out_list.append(self.tsn_time_4(speech, flow_data[3], speech_lengths).unsqueeze(1))

    rgb_out = torch.stack(rgb_out_list, dim=1) if len(rgb_out_list) > 0 else None
    flow_out = torch.stack(flow_out_list, dim=1) if len(flow_out_list) > 0 else None
    # consensus
    rgb_out = self.consensus(rgb_out).squeeze(1) if rgb_out is not None else None
    flow_out = self.consensus(flow_out).squeeze(1) if flow_out is not None else None
    rgb_out = rgb_out.view(rgb_out.shape[0], -1) if rgb_out is not None else None
    flow_out = flow_out.view(flow_out.shape[0], -1) if flow_out is not None else None
    # concat
    output = torch.cat([rgb_out, flow_out], 1) if len(self.modality) == 2 else rgb_out if rgb_out is not None else flow_out
    output = self.relu(output)
    output = self.fc(self.dropout(output))
    return output.squeeze(1)

# test twostream
if __name__ == "__main__":
  model = TSNSEG(2, 64, 4, 'RGB+PA', 'resnet50', 'resnet50', 61, consensus_type='avg', dropout=0.5)
  input = torch.randn(64, 61, 3, 96, 96) # batchsize, length, channel, height, width
  output = model(input)
  print(output.shape)
  print(output)
  print("test twostream pass")