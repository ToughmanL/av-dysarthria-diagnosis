#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 train_test_split.py
* @Time 	:	 2023/03/02
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 在医学领域需要按照人来分测试训练集
'''
import pandas as pd

# 按照ISCSLP 2024 构音障碍音视频分类比赛的划分方式
def dev_train_person():
  dev_persons = ['N_10010_M', 'N_10024_F', 'S_00024_M', 'S_00043_M', 'S_00054_F', 'S_00009_M']
  test_persons = ['N_10007_F', 'N_10004_M', 'N_10012_M', 'N_10015_M', 'S_00032_M', 'S_00070_M', 'S_00062_M', 'S_00046_F', 'S_00045_M', 'S_00008_M', 'S_00003_M', 'S_00012_M', 'S_00013_M', 'S_00021_F']
  train_persons = ['N_10001_F','N_10003_M','N_10004_M','N_10011_M','N_10008_F','N_10009_M','S_00034_M','S_00035_M','S_00044_M','N_10013_M','N_10014_M','N_10023_M','N_10016_M','N_10017_M','N_10018_F','N_10019_M','N_10020_F','N_10021_F','N_10022_F','N_10022_F','S_00004_M','S_00005_M','N_10025_F','S_00014_M','S_00016_M','S_00020_M','S_00022_M','S_00026_M','S_00027_F','S_00030_M','S_00031_F','S_00033_F','S_00047_M','S_00049_F','S_00050_F','S_00051_M','S_00052_F','S_00053_M','S_00055_M','S_00056_F','S_00057_M','S_00058_M','S_00059_M','S_00060_M','S_00061_M','S_00063_M','S_00064_F','S_00065_M','S_00066_F','S_00068_M','S_00069_M', 'S_00023_M','S_00071_M','S_00072_M','S_00073_M','S_00075_M','S_00076_F','S_00077_F','S_00078_M', 'S_00010_F', 'S_00012_M', 'S_00015_F']
  return (dev_persons, test_persons, train_persons)


def _get_minnum(test_folds, fold_data):
  test_min_feat = fold_data.shape[0]
  for test_person in test_folds:
    test_person_num = fold_data[fold_data['Person']==test_person].shape[0]
    if test_person_num < test_min_feat:
      test_min_feat = test_person_num
  return test_min_feat


def data_split(folds, df_data):
  train_list, test_list = [], []
  for test_folds in folds:
    test_pd = pd.DataFrame()
    train_pd = df_data.copy()
    fold_data = df_data.copy()
    for test_person in test_folds:
      person_data = fold_data[fold_data['Person']==test_person]
      # print(test_person, person_data.shape)
      test_pd = pd.concat([test_pd, person_data])
      drop_index = train_pd[train_pd['Person']==test_person].index
      train_pd = train_pd.drop(drop_index)
    train_list.append(train_pd.reset_index(drop=True))
    test_list.append(test_pd.reset_index(drop=True))
  print('data split done')
  return train_list, test_list


def train_test_dev(tr_te_cv, df_data):
  dev_list, test_list, train_list = tr_te_cv
  train_df = df_data[df_data['Person'].isin(train_list)]
  test_df = df_data[df_data['Person'].isin(test_list)]
  dev_df = df_data[df_data['Person'].isin(dev_list)]
  return [train_df], [test_df], [dev_df]



def split_count(df_list):
  for df in df_list:
    print(df['Person'].value_counts())
