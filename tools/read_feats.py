#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 read_feats.py
* @Time 	:	 2024/03/18
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import json
import os
import pandas as pd

# read features from dir
def read_feats(dir_path, suff):
  feat_dict = {}
  for root, dirs, files in os.walk(dir_path, followlinks=True):
    for file in files:
      if file.endswith(suff):
        data_path = os.path.join(root, file)
        ID = file.split('.')[0]
        feat_dict[ID] = data_path
  return feat_dict

# get label
def get_label(label_path='tools/Label.csv'):
  label = pd.read_csv(label_path, encoding='gbk')
  label_dict = label.set_index('Person').T.to_dict()
  return label_dict

# score to label
def score2class(score):
  if score >= 29 and score <= 57:
    return 3
  elif score >= 58 and score <= 86:
    return 2
  elif score >= 87 and score <= 115:
    return 1
  elif score == 116:
    return 0

# convert dict to dataframe
def dict2dataframe(feat_dict, label_name):
  label_dict = get_label()
  listofdict = []
  for id, path in feat_dict.items():
    item = id.replace('repeat_', '')
    nf_num = item.split('_')[0:3]
    person = nf_num[0] + '_' + nf_num[2] + '_' + nf_num[1]
    if label_name == 'classification': # classification
      score = label_dict[person]['Frenchay'] 
      label = score2class(score)
    else: # regression
      label = label_dict[person][label_name] 
    listofdict.append({'Person':person ,'ID': id, 'path': path, label_name: label})
  df = pd.DataFrame(listofdict)
  return df

# dataframe downsample
def dataframe_downsample(dataframe, sample_rate=1):
  if sample_rate == 1:
    return dataframe.sample(frac=1.0, random_state=1).reset_index(drop=True)
  persons = dataframe['Person'].unique()
  result_list = []
  for p in persons:
    person_data = dataframe[dataframe['Person'] == p].reset_index(drop=True)
    down_sample_data = person_data.sample(frac=sample_rate, random_state=1).reset_index(drop=True)
    result_list.append(down_sample_data)
  result_data = pd.concat(result_list, ignore_index=True).reset_index(drop=True)
  return result_data.sample(frac=1.0, random_state=1).reset_index(drop=True)

# read features
def read_features(datapath, feat_type, label_name='classification'):
  feat_data = None
  if feat_type == 'vhubert':
    feat_data = read_feats(datapath, '.npy')
  elif feat_type == 'cmlrv':
    feat_data = read_feats(datapath, '.pt')
  else:
    print('feat_type not supported')
    return None
  df = dict2dataframe(feat_data, label_name)
  return df

# test
if __name__ == '__main__':
  feat_type = 'vhubert'
  datapath = 'vhubert_feats.json'
  task_type = 'classification'
  df = read_features(feat_type, datapath, task_type)
  print(df.head())
  print(df.head())
  print(df.shape)
  print(df['label'].value_counts())
  print(df['Person'].value_counts())
  print(df['ID'].value_counts())
  print(df['path'])