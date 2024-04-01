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

from tools.file_process import mkdir
from tools.extract_args import get_args
from tools.normalization import Normalization
from tools.evaluation import regression_eval, classification_eval
from tools.read_feats import read_features, dataframe_downsample
from tools.train_test_split import normal_test_person, data_split

from networks.model.resnet import ResNetStft
from networks.utils.trainer import NNTrainer
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

# write list of dict to csv file
def write_csv(data, file_name):
  df = pd.DataFrame(data)
  df.to_csv(file_name)

class BaselineRegression():
  def __init__(self, args, test_flag=True):
    # read config
    with open(args.config, 'r') as fin:
      self.configs = yaml.load(fin, Loader=yaml.FullLoader)
    self.feat_type = self.configs['feat_type']
    self.task_info = self.configs['task_info']
    self.task_type,self.label_name,self.label_score = self.task_info['type'], self.task_info['label_name'], self.task_info['score']

    self.test_flag = test_flag
    self.train_list, self.test_list = [], []
    self.fold_list = [int(x) for x in args.fold]
    self.result_dir = f"{self.configs['model_dir']}/{self.feat_type}_{str(self.configs['sample_rate'])}"
    
    # 多卡
    if not ddp:
      mkdir(self.configs['model_dir'] + '/models/')
      mkdir(self.configs['model_dir'] + '/pics/')
      self.device = torch.device('cuda:{}'.format(str(args.gpu)))
    else:
      self.device = device

  # feature normalization
  def _normalization(self, raw_acu_feats):
    NORM = Normalization(self.configs.get('feat_type'))
    assert NORM is not None, 'No such normalization method!'
    norm_acu_feats = NORM.class_normalization(raw_acu_feats) # 91 -9
    # 填充、抽取
    norm_data = NORM.fill_non(norm_acu_feats).dropna().reset_index(drop=True)
    return norm_data

  def _read_feats(self):
    raw_acu_feats = read_features(self.configs.get('data_path'), self.configs.get('feat_type'), self.label_name)
    df_feat = dataframe_downsample(raw_acu_feats, self.configs.get('sample_rate'))
    print('Read feats finised')
    return df_feat

  def _plot_regressor(self, fname, ref, hyp):
    # plt.figure()
    plt.plot(np.arange(len(ref)), ref,'go-',label='true value')
    plt.plot(np.arange(len(ref)),hyp,'ro-',label='predict value')
    plt.title(os.path.basename(fname).split('.')[0])
    plt.legend()
    # plt.show()
    plt.savefig(fname, dpi=120, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None, metadata=None)
    plt.clf()

  def _save_result(self, data, label_name):
    model_path = self.configs['model_dir'] + label_name + '_value.csv'
    data.to_csv(model_path, index=False)

  def _train_test_split(self, df_feat):
    folds = normal_test_person()
    self.train_list, self.test_list = data_split(folds, df_feat)

  def _pad_batch(self, data, batch_size):
    pad_size = batch_size - (len(data) % batch_size)
    pad_data = data.sample(n=pad_size, random_state=1)
    return pd.concat([data, pad_data]).reset_index(drop=True)

  def _model_train(self, dmodel, fold_i):
    label_name, model_name = dmodel['label_name'], dmodel['model_name']
    BatchSize = self.configs['BATCH_SIZE']
    train, test = self.train_list[fold_i], self.test_list[fold_i]
    train = self._pad_batch(train, BatchSize)
    test = self._pad_batch(test, BatchSize)
    val = test.copy()
    Y_test = test[label_name]

    model, _ = init_model(self.configs)
    # write_csv(test, 'data/test.csv')
    val_loader, test_loader, train_loader = load_data(train, test, val, label_name, self.configs, self.test_flag)

    model.to(self.device)
    print(model)
    if ddp:
      model = DDP(model, device_ids=[self.device], find_unused_parameters=True)

    epoch = self.configs['EPOCHS']
    if self.task_type == 'classification':
      loss_function = focal_loss(device=self.device)
    elif self.task_type == 'regression':
      loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=self.configs['LEARNING_RATE'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)

    Trainer = NNTrainer(self.configs, label_name, model, loss_function, optimizer, scheduler)
    if not self.test_flag:
      Trainer.train(train_loader, val_loader, self.device, fold_i)

    # best model test
    model_path = f"{self.configs['model_dir']}/models/{label_name}_fold_{str(fold_i)}_best.pt"
    infos = load_checkpoint(model, model_path)

    # if False:
    #   # 获取中间值
    #   middle_value = Trainer.get_middle_layer(test_loader, 'linear3', self.device)
    #   middle_value.to_csv("wav2vec_Frenchay_fold2.csv")
    #   exit()
    # 测试
    hpy_list, ref_list = Trainer.evaluate(test_loader, self.device)

    X_test_predict = np.vstack(hpy_list)
    X_test_predict = X_test_predict.reshape(-1, 1)

    # regression 中超出范围的分数
    if self.task_type != 'classification':
      X_test_predict[X_test_predict < 0] = 0
      X_test_predict[X_test_predict > self.label_score[label_name]] = self.label_score[label_name]

    if len(test) > len(X_test_predict):
      test = test.iloc[:len(X_test_predict),:]
      Y_test = Y_test.iloc[:len(X_test_predict)]
    sylla_new = pd.DataFrame(test[label_name], columns=[label_name])
    sylla_new = pd.concat([sylla_new, pd.DataFrame(X_test_predict, columns=['Predict'])], axis=1)
    syllabel_test = pd.concat([test.iloc[:,0], sylla_new], axis=1)

    person_test = pd.DataFrame()
    # Subject level (test)
    for person in test['Person'].unique():
      person_index = test['Person'].isin([person])
      person_tmp = pd.DataFrame({'Person': person, label_name: np.mean(Y_test[person_index]), 'Predict': np.mean(X_test_predict[person_index])}, index=[1])
      person_test = person_test.append(person_tmp, ignore_index=True)

    if not self.test_flag:
      result_info = {'name':model_name, 'label':label_name, 'foild':fold_i, 'countlevel':'syllable'}
      if self.task_type == 'regression':
        result_info.update(regression_eval(syllabel_test[label_name], syllabel_test['Predict']))
      elif self.task_type == 'classification':
        result_info.update(classification_eval(syllabel_test[label_name], syllabel_test['Predict']))
      print(result_info)
    return syllabel_test, person_test
  
  def _corss_vali(self, model_label):
    label_name, model_name = model_label['label_name'], model_label['model_name']
    person_pred_total = pd.DataFrame()
    syllabel_pred_total = pd.DataFrame()
    for fold_i in self.fold_list:
      syllabel_test, person_test = self._model_train(model_label, fold_i)
      person_pred_total = pd.concat([person_pred_total, person_test], axis=0)
      syllabel_pred_total = pd.concat([syllabel_pred_total, syllabel_test], axis=0)

    if self.test_flag:
      self._save_result(syllabel_pred_total, label_name)
    syllabel_pred_total = syllabel_pred_total.sort_values(by=['Person']).reset_index(drop=True)
    person_pred_total = person_pred_total.sort_values(by=['Person']).reset_index(drop=True)

    syll_result_info = {'name':model_name, 'label':label_name, 'foild':fold_i, 'countlevel':'syllable'}
    person_result_info = {'name':model_name, 'label':label_name, 'foild':fold_i, 'countlevel':'person'}
    if self.task_type == 'regression':
      syll_result_info.update(regression_eval(syllabel_pred_total[label_name], syllabel_pred_total['Predict']))
      person_result_info.update(regression_eval(person_pred_total[label_name], person_pred_total['Predict']))
    elif self.task_type == 'classification':
      syll_result_info.update(classification_eval(syllabel_pred_total[label_name], syllabel_pred_total['Predict']))
      person_result_info.update(classification_eval(person_pred_total[label_name].round(0).astype(np.int32), person_pred_total['Predict'].round(0).astype(np.int32)))
    print(syll_result_info)
    print(person_result_info)

    # 保存result信息
    self._plot_regressor(self.result_dir + '/Person' + label_name + '_' + model_name + '.png', person_pred_total[label_name], person_pred_total['Predict'])
    self._plot_regressor(self.result_dir + '/syllabel' + label_name + '_' + model_name + '.png', syllabel_pred_total[label_name], syllabel_pred_total['Predict'])
    return person_result_info

  def compute_result(self):
    df_feat = self._read_feats()
    mkdir(self.result_dir)
    self._train_test_split(df_feat)

    results = []
    dmodel = {'label_name':self.label_name, 'model_name':self.configs['model_name']}
    results.append(self._corss_vali(dmodel))
    
    df_result = pd.DataFrame(results)
    csv_path = f"{self.configs['model_dir']}/acoustic_{self.configs['model_name']}_{self.configs['feat_type']}_{str(self.configs['sample_rate'])}.csv"
    df_result.to_csv(csv_path, mode='a')

if __name__ == "__main__":
  args = get_args() # read args
  print(args)
  AR = BaselineRegression(args, args.test_flag)
  AR.compute_result()
  print("Done!")

# python local/main.py --config conf/base_config.yaml --gpu 0 --fold ['0'] --test_flag 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25641 baseline_regression.py