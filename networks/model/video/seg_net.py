import torch
from torch import nn


class SegNet(nn.Module):
  def __init__(self, num_segments):
    super(SegNet, self).__init__()
    if num_segments == -4: # 按照构音障碍来分段
      self.segments_len = [10, 15, 29, 7]
    else: # 按照num_segments来分段
      base_len = new_length // num_segments
      add_len = new_length % num_segments
      self.segments_len = [base_len + 1 if i < add_len else base_len for i in range(num_segments)]
  
  # random select index
  def select_index(self, data, num_segments, rgb_len=1):
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
  
  def forward(
        self,
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lengths: torch.Tensor
    ):
    input = video
    segment_num = len(self.segments_len)
    if segment_num == 4: # 包括构音障碍划分和随机划分
      # 将数据分为四段，第一段为静音段10帧，第二段为AIT段15帧，第四段为后段7帧，中间段为剩余帧
      front_len, ait_len, middle_len, back_len = self.segments_len
      front_data = input[:, :front_len]
      ait_data = input[:, front_len:front_len + ait_len]
      middle_data = input[:, front_len + ait_len:-back_len]
      back_data = input[:, -back_len:]
      input_data = [front_data, ait_data, middle_data, back_data]
      return self.select_index(input_data, segment_num)
    else: # 随机划分
      data_list = []
      start = 0
      for i in range(segment_num):
        data_list.append(input[:, start:start+self.segments_len[i]])
        start += self.segments_len[i]
      return self.select_index(data_list, segment_num)
