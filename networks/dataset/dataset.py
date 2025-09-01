#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 dataset.py
* @Time 	:	 2023/02/23
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import random
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import IterableDataset

from sklearn.preprocessing import StandardScaler
from networks.dataset import data_processor
from networks.dataset.transforms import process_tensor_list
from networks.utils.cmvn import load_cmvn


# 将列表分为四段，第一段为静音段10帧，第二段为AIT段15帧，第四段为后段7帧，中间段为剩余帧
def split_data(data, front_len=10, ait_len=15, back_len=7):
  data_len = len(data)
  if data_len < front_len + ait_len + back_len:
    raise ValueError("Data length is not enough")
  front_data = data[:front_len]
  ait_data = data[front_len:front_len+ait_len]
  middle_data = data[front_len+ait_len-1:-back_len]
  back_data = data[-back_len:]
  return front_data, ait_data, middle_data, back_data


def RNN_collate(data_y_tuple):
  train_data = [item[0] for item in data_y_tuple]
  label = torch.tensor([item[1] for item in data_y_tuple])
  train_data.sort(key=lambda data: len(data), reverse=True)
  # data_length = [len(data) for data in train_data]
  train_data = pad_sequence(train_data, batch_first=True, padding_value=0.0)
  return train_data, label


def TSN_collate(data_y_tuple):
  # 分离前段和后段，并按要求进行填充和拼接
  front_list, ait_list, middle_list, back_list = [], [], [], []
  labels = []

  for item in data_y_tuple:
    data = item[0]
    label = item[1]
    front_data, ait_data, middle_data, back_data = split_data(data)
    front_list.append(front_data)
    ait_list.append(ait_data)
    middle_list.append(middle_data)
    back_list.append(back_data)
    labels.append(label)

  # 对前段数据进行 padding
  middle_padded = process_tensor_list(middle_list)
  front_padded = torch.stack(front_list)
  ait_padded = torch.stack(ait_list)
  back_padded = torch.stack(back_list)
  # 将前段数据和填充后段数据拼接起来
  combined_data = torch.cat((front_padded, ait_padded, middle_padded, back_padded), dim=1)

  # 创建标签张量
  label_tensor = torch.tensor(labels)

  return combined_data, label_tensor


class Processor(IterableDataset):
  def __init__(self, source, f, *args, **kw):
      assert callable(f)
      self.source = source
      self.f = f
      self.args = args
      self.kw = kw

  def set_epoch(self, epoch):
      self.source.set_epoch(epoch)

  def __iter__(self):
      """ Return an iterator over the source dataset processed by the
        given processor.
      """
      assert self.source is not None
      assert callable(self.f)
      return self.f(iter(self.source), *self.args, **self.kw)

  def apply(self, f):
      assert callable(f)
      return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
  def __init__(self, shuffle=True, partition=True):
      self.epoch = -1
      self.update()
      self.shuffle = shuffle
      self.partition = partition

  def update(self):
      assert dist.is_available()
      if dist.is_initialized():
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
      else:
        self.rank = 0
        self.world_size = 1
      worker_info = torch.utils.data.get_worker_info()
      if worker_info is None:
        self.worker_id = 0
        self.num_workers = 1
      else:
        self.worker_id = worker_info.id
        self.num_workers = worker_info.num_workers
      return dict(rank=self.rank,
              world_size=self.world_size,
              worker_id=self.worker_id,
              num_workers=self.num_workers)

  def set_epoch(self, epoch):
      self.epoch = epoch

  def sample(self, data):
      data = list(range(len(data)))
      if self.partition:
        if self.shuffle:
          random.Random(self.epoch).shuffle(data)
        data = data[self.rank::self.world_size]
      data = data[self.worker_id::self.num_workers]
      return data


# command data loader
class NNDataLoder(torch.utils.data.Dataset):
  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.tensor(X.values.astype(np.float32))
      self.y = torch.tensor(y.values.astype(np.float32))

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]


class DataList(IterableDataset):
  def __init__(self, lists, shuffle=True, partition=True):
    self.lists = lists
    self.sampler = DistributedSampler(shuffle, partition)

  def set_epoch(self, epoch):
    self.sampler.set_epoch(epoch)

  def __iter__(self):
    sampler_info = self.sampler.update()
    indexes = self.sampler.sample(self.lists)
    for index in indexes:
      data = dict(src=self.lists[index])
      data.update(sampler_info)
      yield data


def IterDataset(data, conf):
  audio_feat = conf.get('audiofeat', None)
  video_feat = conf.get('videofeat', None)
  text_feat = conf.get('textfeat', None)
  batch_size = conf.get('batch_size', 8)
  raw_dict_list = list(data.T.to_dict().values())
  dataset = DataList(raw_dict_list, False, False)

  dataset = Processor(dataset, data_processor.audio_videl_feat, audio_feat, video_feat, text_feat)
  if audio_feat != None:
    # cmvn
    if audio_feat == 'fbank' and 'cmvn_file' in conf:
      mean, istd = load_cmvn(conf['cmvn_file'], True)
      dataset = Processor(dataset, data_processor.cmvn, torch.from_numpy(mean).float(), torch.from_numpy(istd).float())

  auduiseg_num = conf.get('audiosegnum', 0)
  # padding and batch
  a_num_frms = conf.get('a_num_frms', 0)
  v_num_frms = conf.get('v_num_frms', 0)
  num_token = conf.get('num_token', 0)
  dataset = Processor(dataset, data_processor.batch, batch_size)
  dataset = Processor(dataset, data_processor.padding, a_num_frms, v_num_frms, num_token, auduiseg_num)
  return dataset


if __name__ == "__main__":
  import time
  import yaml

  start_time = time.time()
  config_file = 'conf/gopd_linear.yaml'
  csv_path = 'data/filtered_test.csv'
  raw_acu_feats = pd.read_csv(csv_path)
  with open(config_file, 'r') as fin:
    conf = yaml.load(fin, Loader=yaml.FullLoader)

  base_dataset = IterDataset(raw_acu_feats, conf)
  data_loader = torch.utils.data.DataLoader(base_dataset, batch_size=None, num_workers=0)
  for i, batch in enumerate(data_loader):
    print(len(batch))
    key, audio_feat, video_feat, text_feat, target, audio_len, text_len = batch
    print(audio_feat)
    print(audio_feat.size())
    print(video_feat.size())
    print(target.size())
    if i == 10:
      break
  
  end_time = time.time()
  print("duraion: {:.2f}sesonds".format(end_time - start_time))

  # fbank size = (B, T, 80)