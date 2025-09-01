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
from sklearn.utils import resample

# read features from dir
def read_feats(dir_path, suff):
  feat_dict = {}
  # 判断是否为文件夹或者是文件
  if os.path.isdir(dir_path):
    for root, dirs, files in os.walk(dir_path, followlinks=True):
      for file in files:
        if file.endswith(suff):
          data_path = os.path.join(root, file).split('.')[0]
          ID = file.split('.')[0]
          feat_dict[ID] = data_path
  elif os.path.isfile(dir_path):
    with open(dir_path, 'r') as f:
      raw_feat_dict = json.load(f)
    for ID, path in raw_feat_dict.items():
      new_path = path.split('.')[0]
      if not os.path.exists(new_path+suff):
        print('path not exists:', path)
        continue
      feat_dict[ID] = new_path
  return feat_dict

# get label
def get_label(label_path='tools/MSDMLabel.csv'):
  label = pd.read_csv(label_path, encoding='gbk')
  label_dict = label.set_index('Person').T.to_dict()
  return label_dict

# score to label
def score2class(score, num_class, label_name):
  if label_name == 'Frenchay':
    if num_class == 2:
      if score == 116:
          return 0
      elif score < 116:
        return 1
    elif num_class == 4:
      if score >= 29 and score <= 57:
        return 3
      elif score >= 58 and score <= 86:
        return 2
      elif score >= 87 and score <= 115:
        return 1
      elif score == 116:
        return 0
    else:
      ValueError('num_class not supported')
  elif label_name == 'NumberA':
    if num_class == 2:
      if score < 27:
        return 1
      else:
        return 0
    elif num_class == 4:
      if score >= 0 and score <= 6:
        return 3
      elif score >= 7 and score <= 12: # 7-17
        return 2
      elif score >= 13 and score <= 28: # 20-26
        return 1
      elif score >= 29: # 27-29
        return 0
    else:
      ValueError('num_class not supported')

# convert dict to dataframe
def msdmdict(feat_dict, num_class, label_name, labelfile):
  label_dict = get_label(labelfile)
  listofdict = []
  for id, path in feat_dict.items():
    item = id.replace('repeat_', '')
    nf_num = item.split('_')[0:3]
    person = nf_num[0] + '_' + nf_num[2] + '_' + nf_num[1]
    if label_name == 'classification': # classification
      score_name = 'Frenchay' # 'NumberA' or 'Frenchay'
      score = label_dict[person][score_name] 
      label = score2class(score, num_class, score_name)
    else: # regression
      label = label_dict[person][label_name] 
    listofdict.append({'Person':person ,'ID': id, 'path': path, 'label': label})
  df = pd.DataFrame(listofdict)
  return df

def uaspeechdict(feat_dict, num_class, label_name, labelfile):
  pass

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

# classes balance
def balance_classes(df):
  class_counts = df['label'].value_counts() # 计算每个类别的样本数量
  target_samples = class_counts.max()
  # target_samples = class_counts.min()
  # target_samples = int(class_counts.mean())

  balanced_df = pd.DataFrame(columns=df.columns)
  for label in class_counts.index:
    label_df = df[df['label'] == label] # 获取当前类别的数据

    if len(label_df) < target_samples:
      n_samples = int(target_samples - len(label_df))
      resampled_df = resample(label_df, replace=True, n_samples=n_samples, random_state=42)
      # 将升采样后的数据加入平衡后的DataFrame中
      balanced_df = pd.concat([balanced_df, label_df, resampled_df])
    else:
      # 如果当前类别的样本数量大于等于目标样本数量，抽取目标样本数量的样本
      sample_df = label_df.sample(n=target_samples, random_state=42)
      balanced_df = pd.concat([balanced_df, sample_df])

  balanced_df.reset_index(drop=True, inplace=True)
  return balanced_df

# get classes freqency
def get_freq(df):
  # 计算从小到大的每个类别的样本数量
  class_counts = df['label'].value_counts()
  class_freq = class_counts/len(df)
  return 1- class_freq

# read features
def read_features(datapath, audio_feat, video_feat, text_feat, num_class, labelfile, label_name='classification'):
  feat_data = None
  pt_name = ['crossatt', 'decoder_dist', 'encoderout', 'text_dist', 'textemb']
  huberts = ['vhubert', 'ahubert', 'avhubert']
  wavs = ['wav', 'fbank', 'mfcc', 'melspec']
  if audio_feat in pt_name :
    feat_data = read_feats(datapath, '.pt')
  elif audio_feat in huberts or video_feat in huberts:
    feat_data = read_feats(datapath, '.npy')
  elif 'cropvideo' in video_feat:
    feat_data = read_feats(datapath, '.pt')
  elif audio_feat in wavs:
    feat_data = read_feats(datapath, '.wav')
  elif 'wav2vec' in audio_feat:
    feat_data = read_feats(datapath, '.wav2vec.pt')
  elif audio_feat == 'ivector':
    feat_data = read_feats(datapath, '.ivector.pt')
  elif audio_feat == 'egemaps':
    feat_data = read_feats(datapath, '.egemaps.pt')
  elif video_feat == 'cmlrv':
    feat_data = read_feats(datapath, '.cmlrv.pt')
  elif text_feat == 'gop':
    feat_data = read_feats(datapath, '.gop.pt')
  elif text_feat == 'gopd':
    feat_data = read_feats(datapath, '.gopd.pt')
  else:
    print('feat_type not supported')
    return None
  if 'MSDM' in labelfile:
    df = msdmdict(feat_data, num_class, label_name, labelfile)
  elif 'UASpeech' in labelfile:
    df = uaspeechdict(feat_data, num_class, label_name, labelfile)
  return df

# test
if __name__ == '__main__':
  import yaml
  audiofeat = 'fbank'
  textfeat = ''
  videofeat= ''
  label_name = 'classification'
  label_file = 'tools/MSDMLabel.csv'
  config_file='conf/audio_resnetseq10_300f.yaml'
  with open(config_file, 'r') as fin:
      configs = yaml.load(fin, Loader=yaml.FullLoader)
  df = read_features(configs.get('data_path'), audiofeat, videofeat, textfeat, configs['task_info']['score'], label_file, label_name)
  print(df.head())
  print(df.shape)
  print(df['label'].value_counts())
  print(df['Person'].value_counts())
  print(df['ID'].value_counts())
  print(df['path'])