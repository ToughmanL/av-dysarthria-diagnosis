#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 load_data.py
* @Time 	:	 2024/03/19
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
from torch.utils.data import DataLoader
from networks.dataset.dataset import IterDataset, NNDataLoder, RNN_collate

def load_data(train, test, val, label, config, test_flag):
  feat_type = config['feat_type']
  BatchSize = config['BATCH_SIZE']
  feat_dict, cmvn = {}, {}
  val_loader, test_loader, train_loader = None, None, None
  if config['dataloader'] == 'IterDataset':
    test_loader = DataLoader(IterDataset(test, feat_type, label), collate_fn=RNN_collate, batch_size=BatchSize, num_workers=16, drop_last=True)
    if not test_flag: # 默认shuffle=False
      val_loader = test_loader
      train_loader = DataLoader(IterDataset(train, feat_type, label), collate_fn=RNN_collate, batch_size=BatchSize, num_workers=16, drop_last=True)
  elif config['dataloader'] == 'NNDataLoder':
    X_train, Y_train = train.iloc[:,1:-1], train[label]
    X_val, Y_val = val.iloc[:,1:-1], val[label]
    X_test, Y_test = test.iloc[:,1:-1], test[label]
    val_loader = DataLoader(NNDataLoder(X_val, Y_val), batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=16)
    test_loader = DataLoader(NNDataLoder(X_test, Y_test), batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=16)
    train_loader = DataLoader(NNDataLoder(X_train, Y_train), batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=16)
  else:
    raise ValueError("dataloader not supported")

  return val_loader, test_loader, train_loader


if __name__ == '__main__':
  import pandas as pd
  train = pd.read_csv('data/test.csv')
  test = pd.read_csv('data/test.csv')
  val = pd.read_csv('data/test.csv')
  label = 'classification'
  config = {
    'feat_type': 'vhubert',
    'BATCH_SIZE': 2,
    'dataloader': 'IterDataset'
  }
  val_loader, test_loader, train_loader = load_data(train, test, val, label, config, False)
  for x, y in train_loader:
    print(x.size())
    print(y.size())
    break
  print('done')