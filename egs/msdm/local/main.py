#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 regression.py
* @Time 	:	 2024/03/18
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

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
from tools.evaluation import regression_eval, classification_eval, classification_eval_person
from tools.read_feats import read_features, dataframe_downsample, balance_classes, get_freq
from tools.train_test_split import dev_train_person, train_test_dev

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

  # plot the regression result
  def _plot_regressor(self, fname, ref, hyp):
    # plt.figure()
    plt.plot(np.arange(len(ref)), ref,'go-',label='true value')
    plt.plot(np.arange(len(ref)),hyp,'ro-',label='predict value')
    plt.title(os.path.basename(fname).split('.')[0])
    plt.legend()
    # plt.show()
    plt.savefig(fname, dpi=120, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None, metadata=None)
    plt.clf()

  # save the result to csv file
  def _save_result(self, data, label_name):
    model_path = self.configs['model_dir'] + label_name + '_value.csv'
    data.to_csv(model_path, index=False)

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
    BatchSize = self.configs['batch_size']
    train, test, dev = self.train_list[fold_i], self.test_list[fold_i], self.dev_list[fold_i]
    train = self._pad_batch(train, BatchSize)
    test = self._pad_batch(test, BatchSize)
    val = self._pad_batch(dev, BatchSize)
    Y_test = test['label']
    model, _ = init_model(self.configs, fold_i)
    val_loader, test_loader, train_loader = load_data(train, test, val, self.configs, self.test_flag)

    model.to(self.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"总参数量 (M): {total_params / 1e6:.2f}M")

    if ddp:
      model = DDP(model, device_ids=[self.device], find_unused_parameters=True)

    epoch = self.configs['EPOCHS']
    if self.task_type == 'classification':
      weithts = get_freq(train)
      weights_tensor = torch.tensor(weithts, dtype=torch.float32).to(self.device)
      loss_function = focal_loss(device=self.device)
    elif self.task_type == 'regression':
      loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=self.configs['LEARNING_RATE'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)

    train_type = self.configs.get('train_type', 'speech')
    if train_type == 'speech':
      Trainer = SpeechTrainer(self.configs, label_name, model, loss_function, optimizer, scheduler)
    elif train_type == 'NN':
      Trainer = NNTrainer(self.configs, label_name, model, loss_function, optimizer, scheduler)
    elif train_type == 'MTLADD':
      Trainer = MTLAddTrainer(self.configs, label_name, model, loss_function, optimizer, scheduler)
    elif train_type == 'MTLMMP':
      Trainer = MTLMMPTrainer(self.configs, label_name, model, loss_function, optimizer, scheduler)
    else:
      raise ValueError('train_type error')

    if not self.test_flag:
      Trainer.train(train_loader, val_loader, self.device, fold_i)

    # best model test
    model_path = f"{self.configs['model_dir']}/models/{label_name}_fold_{str(fold_i)}_best.pt"
    infos = load_checkpoint(model, model_path)

    # 测试
    key_list, hpy_list, ref_list, last_feat = Trainer.evaluate(test_loader, self.test_flag, self.device)
    key_array = np.array(key_list)
    person_list = [id2person(id) for id in key_list]
    person_array = np.array(person_list)
    ref_array = np.array(ref_list).flatten()
    X_test_predict = np.array(hpy_list)
    last_feat_array = np.array(last_feat)

    syllabel_test = pd.DataFrame({"ID":key_array, "Person":person_array, label_name: ref_array, "Predict": X_test_predict})

    person_test = pd.DataFrame()
    # Subject level (test)
    for person in test['Person'].unique():
      ref_person_index = test['Person'].isin([person])
      ref_mode_score = test['label'][ref_person_index].mode().values[0]
      phy_person_index = syllabel_test['Person'].isin([person])
      phy_mode_score = syllabel_test['Predict'][phy_person_index].mode().values[0]
      person_tmp = pd.DataFrame({'Person': person, label_name: ref_mode_score, 'Predict': phy_mode_score}, index=[1])
      person_test = person_test.append(person_tmp, ignore_index=True)

    result_info = {'name':model_name, 'label':label_name, 'foild':fold_i, 'countlevel':'syllable'}
    if self.task_type == 'regression':
      result_info.update(regression_eval(syllabel_test[label_name], syllabel_test['Predict']))
    elif self.task_type == 'classification':
      result_info.update(classification_eval(syllabel_test[label_name], syllabel_test['Predict']))
    print(result_info)

    # 判断last_feat_array是否为空
    if len(last_feat_array) > 1:
      x = last_feat_array.shape[1]
      feature_columns = [f'feat_{i}' for i in range(x)]
      combined_data = np.hstack([last_feat_array, key_array[:, np.newaxis], person_array[:, np.newaxis], X_test_predict[:, np.newaxis], ref_array[:, np.newaxis]])  # 最终形状 (11392, 8)
      column_names = feature_columns + ['Key', 'Person', 'Predict', 'Target']
      last_feat_test = pd.DataFrame(combined_data, columns=column_names)
    return syllabel_test, person_test, last_feat_test
  
  def _corss_vali(self, model_label):
    label_name, model_name = model_label['label_name'], model_label['model_name']
    person_pred_total = pd.DataFrame()
    syllabel_pred_total = pd.DataFrame()
    last_feat_total = pd.DataFrame()
    for fold_i in self.fold_list:
      syllabel_test, person_test, last_feat_array = self._model_train(model_label, fold_i)
      person_pred_total = pd.concat([person_pred_total, person_test], axis=0)
      syllabel_pred_total = pd.concat([syllabel_pred_total, syllabel_test], axis=0)
      # 判断last_feat_array是否为空
      if len(last_feat_array) > 1:
        last_feat_total = pd.concat([last_feat_total, last_feat_array], axis=0)

    if self.test_flag:
      self._save_result(syllabel_pred_total, label_name)
    syllabel_pred_total = syllabel_pred_total.sort_values(by=['Person']).reset_index(drop=True)
    person_pred_total = person_pred_total.sort_values(by=['Person']).reset_index(drop=True)

    syll_result_info = {'name':model_name, 'label':label_name,'countlevel':'syllable'}
    person_result_info = {'name':model_name, 'label':label_name, 'countlevel':'person'}
    each_person_acc = {}
    if self.task_type == 'regression':
      syll_result_info.update(regression_eval(syllabel_pred_total[label_name], syllabel_pred_total['Predict']))
      person_result_info.update(regression_eval(person_pred_total[label_name], person_pred_total['Predict']))
    elif self.task_type == 'classification':
      syll_result_info.update(classification_eval(syllabel_pred_total[label_name], syllabel_pred_total['Predict']))
      person_result_info.update(classification_eval(person_pred_total[label_name].round(0).astype(np.int32), person_pred_total['Predict'].round(0).astype(np.int32)))
      each_person_acc.update(classification_eval_person(syllabel_pred_total, label_name))
    print(syll_result_info)
    print(person_result_info)
    for key, value in each_person_acc.items():
      print(key, value)

    # 保存result信息
    self._plot_regressor(self.result_dir + '/Person' + label_name + '_' + model_name + '.png', person_pred_total[label_name], person_pred_total['Predict'])
    self._plot_regressor(self.result_dir + '/syllabel' + label_name + '_' + model_name + '.png', syllabel_pred_total[label_name], syllabel_pred_total['Predict'])

    if len(last_feat_total) > 1:
      last_feat_total.to_csv(self.result_dir + '/last_feat_total.csv', index=False)
    return person_result_info

  def compute_result(self):
    df_feat = self._read_feats()
    mkdir(self.result_dir)
    self._train_test_split(df_feat)

    results = []
    dmodel = {'label_name':self.label_name, 'model_name':self.configs['model_name']}
    results.append(self._corss_vali(dmodel))
    
    df_result = pd.DataFrame(results)
    csv_path = f"{self.configs['model_dir']}/acoustic_{self.configs['model_name']}_{self.feat_type}_{str(self.configs['sample_rate'])}.csv"
    df_result.to_csv(csv_path, mode='a')

if __name__ == "__main__":
  args = get_args() # read args
  print(args)
  AR = BaselineRegression(args, args.test_flag)
  AR.compute_result()
  print("Done!")
