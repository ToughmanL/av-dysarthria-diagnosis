#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
* @File 	:	 compute_param_flops.py
* @Time 	:	 2025/08/24
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import torch
import torch.nn as nn
import time
from thop import profile, clever_format

import os
import yaml
import torch
import random
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tools.file_process import mkdir, id2person
from tools.extract_args import get_args
# from tools.normalization import Normalization
from tools.evaluation import regression_eval, classification_eval, classification_eval_person
from tools.read_feats import read_features, dataframe_downsample, balance_classes, get_freq
from tools.train_test_split import normal_test_person, dev_train_person, data_split, train_test_dev

# from networks.model.resnet import ResNetStft
from networks.utils.trainer import NNTrainer, SpeechTrainer, MTLAddTrainer, MTLMMPTrainer
from networks.utils.init_model import init_model
from networks.dataset.load_data import load_data
from networks.utils.loss_fun import focal_loss
from networks.utils.checkpoint import load_checkpoint



import warnings
warnings.filterwarnings('ignore')

ddp = False
if ddp:
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP
  dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1800))
  local_rank = dist.get_rank()
  torch.cuda.set_device(local_rank)
  device = torch.device('cuda', local_rank)
  # 固定随机种子
  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
else:
  device = torch.device('cpu')

# write list of dict to csv file
def write_csv(data, file_name):
  df = pd.DataFrame(data)
  df.to_csv(file_name)

class BaselineRegression():
  def __init__(self, args, test_flag=True):
    # read config
    with open(args.config, 'r') as fin:
      self.configs = yaml.load(fin, Loader=yaml.FullLoader)
    self.audiofeat = self.configs.get('audiofeat', '')
    self.videofeat = self.configs.get('videofeat', '')
    self.textfeat = self.configs.get('textfeat', '')
    self.feat_type = self.audiofeat + self.videofeat
    self.task_info = self.configs['task_info']
    self.task_type,self.label_name,self.label_score = self.task_info['type'], self.task_info['label_name'], self.task_info['score']

    self.test_flag = test_flag
    self.train_list, self.test_list, self.dev_list = [], [], []
    self.fold_list = [int(x) for x in args.fold]
    self.result_dir = f"{self.configs['model_dir']}/{self.feat_type}_{str(self.configs['sample_rate'])}"

    # 多卡
    if not ddp:
      mkdir(self.configs['model_dir'] + '/models/')
      mkdir(self.configs['model_dir'] + '/pics/')
      self.device = torch.device('cuda:{}'.format(str(args.gpu)))
    else:
      self.device = device

  # read features from csv file
  def _read_feats(self):
    label_file = 'tools/MSDMLabel.csv' if 'label_file' not in self.configs else self.configs['label_file']
    raw_acu_feats = read_features(self.configs.get('data_path'), self.audiofeat, self.videofeat, self.textfeat, self.configs['task_info']['score'], label_file, self.label_name)
    df_feat = dataframe_downsample(raw_acu_feats, self.configs.get('sample_rate'))
    print('Dataframe length:', len(df_feat))
    print('Read feats finised')
    return df_feat


  # split the data into train and test
  def _train_test_split(self, df_feat):
    tr_te_cv = dev_train_person()
    self.train_list, self.test_list, self.dev_list = train_test_dev(tr_te_cv, df_feat)
  # pad last batch
  def _pad_batch(self, data, batch_size): 
    pad_size = batch_size - (len(data) % batch_size)
    pad_data = data.sample(n=pad_size, random_state=1)
    return pd.concat([data, pad_data]).reset_index(drop=True)

  def _model_train(self, dmodel, fold_i):
    label_name, model_name = dmodel['label_name'], dmodel['model_name']
    self.configs['batch_size'] = 1 if self.test_flag else self.configs['batch_size']
    BatchSize = self.configs['batch_size']
    train, test, dev = self.train_list[fold_i], self.test_list[fold_i], self.dev_list[fold_i]
    train = self._pad_batch(train, BatchSize)
    test = self._pad_batch(test, BatchSize)
    # val = self._pad_batch(dev, BatchSize)
    val = test.copy()

    Y_test = test['label']

    model, _ = init_model(self.configs, fold_i)
    # write_csv(test, 'data/test.csv')
    val_loader, test_loader, train_loader = load_data(train, test, val, self.configs, self.test_flag)

    model.to(self.device)

    sample_nums = 0

    start = time.perf_counter()

    with torch.no_grad():
      keys, predictions, values, last_feat = [], [], [], []
      for batch_idx, batch in enumerate(test_loader):
        key, audio_feats, video_feats, target, feats_lengths, video_len = batch
        audio_feats = audio_feats.to(self.device)
        video_feats = video_feats.to(self.device)
        target = target.to(self.device)
        feats_lengths = feats_lengths.to(self.device)
        video_len = video_len.to(self.device)
        model.eval()
        last_layer_feat = model(audio_feats, video_feats, feats_lengths)
        # flops, params = profile(model, inputs=(audio_feats, video_feats, feats_lengths), verbose=False)
        # print(f"Fold {fold_i}, Batch {batch_idx}, FLOPs: {flops}, Params: {params}")
        if self.task_type == 'classification':
          yhat = torch.argmax(last_layer_feat, dim=1)
        keys.extend(key)
        predictions.extend(yhat.cpu().detach().tolist())
        values.extend(target.unsqueeze(1).cpu().detach().tolist())
        last_feat.extend(last_layer_feat.cpu().detach().tolist())
        sample_nums += len(key)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"GPU执行时间: {end - start:.6f} 秒")
    print(f"Fold {fold_i}, Sample nums: {sample_nums}")


    # if False:  # 获取模型中间表征
    #   middle_value = Trainer.get_middle_layer(test_loader, 'linear3', self.device)
    #   middle_value.to_csv("wav2vec_Frenchay_fold2.csv")
    #   exit()

  def compute_result(self):
    df_feat = self._read_feats()
    mkdir(self.result_dir)
    self._train_test_split(df_feat)
    results = []
    dmodel = {'label_name':self.label_name, 'model_name':self.configs['model_name']}
    self._model_train(dmodel, 0)
    

if __name__ == "__main__":
  args = get_args() # read args
  print(args)
  AR = BaselineRegression(args, args.test_flag)
  AR.compute_result()
  print("Done!")
