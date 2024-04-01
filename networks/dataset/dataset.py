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

import torch
import torch.distributed as dist
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import IterableDataset

from networks.dataset.data_processor import DataProcessor


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

def RNN_collate(data_y_tuple):
  train_data = [item[0] for item in data_y_tuple]
  label = torch.tensor([item[1] for item in data_y_tuple])
  train_data.sort(key=lambda data: len(data), reverse=True)
  # data_length = [len(data) for data in train_data]
  train_data = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True, padding_value=0)
  return train_data, label

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

def IterDataset(data, feat_type, label, data_dir=None):
  DP = DataProcessor(label, data_dir, 450)
  raw_dict_list = list(data.T.to_dict().values())
  dataset = DataList(raw_dict_list, False, False)
  if feat_type == 'stft':
    dataset = Processor(dataset, DP.compute_stft)
  elif feat_type == 'fbank':
    dataset = Processor(dataset, DP.compute_fbank)
  elif feat_type == 'mfcc':
    dataset = Processor(dataset, DP.compute_mfcc)
  elif feat_type == 'cmlrv':
    dataset = Processor(dataset, DP.get_cmlrv, label=label)
  elif feat_type == 'wav2vec':
    dataset = Processor(dataset, DP.get_wav2vecmax)
  elif feat_type == 'hubert':
    dataset = Processor(dataset, DP.get_hubertmax)
  elif feat_type == 'vhubert':
    dataset = Processor(dataset, DP.get_vhubert, label=label)
  elif feat_type == 'videoflow':
    dataset = Processor(dataset, DP.get_videoflow, label=label)
  else:
    raise ValueError("feat_type not supported")
  return dataset



if __name__ == "__main__":
  import time
  start_time = time.time()
  csv_path = 'data/test.csv'
  raw_acu_feats = pd.read_csv(csv_path)

  base_dataset = IterDataset(raw_acu_feats, 'cmlrv', 'classification')
  data_loader = torch.utils.data.DataLoader(base_dataset, collate_fn=RNN_collate, batch_size=2, num_workers=1)
  # data_loader = DataLoaderX(base_dataset, batch_size=64, num_workers=16)
  for x, y in data_loader:
    print(x.shape, y.shape)
  
  end_time = time.time()
  print("duraion: {:.2f}sesonds".format(end_time - start_time))