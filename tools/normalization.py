#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 normalization.py
* @Time 	:	 2023/02/08
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyself.right 2022-2025, lxk&AISML
* @Desc   	:	 
'''

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',80)

class Normalization():
  def __init__(self, feat_type):
    feat_type_dict = {'loop_featnorm': 'command', 'gop_loop_featnorm': 'command', 'loop_v_featnorm': 'command', 'loop_av_featnorm': 'command', 'gop': 'command', 'Liuartifeat': 'command'}
    if feat_type in  feat_type_dict: # 为所有特征计算标准差
      self.left = -1
      self.right = -9
    else:
      return None

  def class_normalization(self, df_data):
    '''
     @func _normalization
     @desc data normalization
     @param {df_data}  
     @return {df_data} 
    '''
    Compare = pd.DataFrame()
    normal_data = df_data.copy()
    # normal_data = normal_data.drop(['TEXT', 'Vowel'], axis=1)
    X_scale = preprocessing.scale(normal_data.iloc[:, self.left:self.right], axis=0)
    Compare_temp = pd.DataFrame(X_scale, columns=normal_data.iloc[:, self.left:self.right].columns)
    
    # 合并name Person TEXT Frenchay
    Compare_temp = pd.concat([normal_data[['Person']], Compare_temp], axis=1)
    Compare_temp = pd.concat([Compare_temp, normal_data.iloc[:, self.right:]], axis=1)
    Compare = pd.concat([Compare, Compare_temp], axis=0)
    return Compare
  
  def half_normalization(self, df_data):
    '''
     @func _normalization
     @desc data normalization
     @param {df_data}  
     @return {df_data} 
    '''
    compare_data = df_data.copy()
    normal_data = df_data[df_data['Person'].str.contains('N_')].reset_index(drop=True)
    mms = preprocessing.StandardScaler()
    mms.fit(normal_data.iloc[:, self.left:self.right])
    Compare_temp = pd.DataFrame(mms.transform(compare_data.iloc[:, self.left:self.right]),columns=compare_data.iloc[:, self.left:self.right].columns)
    Compare_temp = pd.concat([compare_data[['Person']], Compare_temp, compare_data.iloc[:,self.right:]], axis=1)
    return Compare_temp

  def lxk_common_sylla(self, df_data):
    '''
     @func _normalization_syllable
     @desc 对每种音节中的每种特征进行标准化
     @param {特征文件}
     @return {返回标准化矩阵字典}
    '''
    data = df_data.copy()
    Normal = data[data['Person'].str.contains('N_')].reset_index(drop=True)
    syllable_set = data['TEXT'].unique()
    syll_all_dict = {}
    common_feats = ['Jitter', 'Shimmer', 'HNR', 'gne', 'vfer', 'F1_sd', 'F2_sd', 'F3_sd', 'Intensity_mean', 'Intensity_sd', 'Vowel_dur', 'Syllable_dur']
    # 按音节进行归一化，common特征
    for sylla in syllable_set:
      normal_data = Normal[Normal['TEXT'] == sylla].reset_index(drop=True)
      syll_data = data[data['TEXT'] == sylla].reset_index(drop=True)
      norm_sylla_dict = {}
      for comfeat in common_feats:
        # 使用正常人的数据进行训练
        standar_dict = {}
        comfeat_data = normal_data[comfeat].reset_index(drop=True)
        if len(comfeat_data) == 0:
          comfeat_data = syll_data[comfeat].reset_index(drop=True)
        standar_dict['mean'] = comfeat_data.mean()
        standar_dict['std'] = comfeat_data.std()
        norm_sylla_dict[comfeat] = standar_dict
      syll_all_dict[sylla] = norm_sylla_dict
    return syll_all_dict

  def lxk_arti_feat(self, df_data):
    '''
     @func _normalization
     @desc 对每种六个发音特征进行特折级别归一化
     @param {df_data}  
     @return {df_data} 
    '''
    Compare = pd.DataFrame()
    normal_data = df_data.copy()
    # normal_data = normal_data.drop(['TEXT', 'Vowel'], axis=1)
    X_scale = preprocessing.scale(normal_data.iloc[:, 2:8], axis=0)
    Compare_temp = pd.DataFrame(X_scale, columns=normal_data.iloc[:, 2:8].columns)
    
    # 合并name Person TEXT Frenchay
    Compare_temp = pd.concat([normal_data[['Person']], Compare_temp], axis=1)
    Compare_temp = pd.concat([Compare_temp, normal_data.iloc[:, 8:]], axis=1)
    Compare = pd.concat([Compare, Compare_temp], axis=0)
    return Compare
  
  def class_normalization_x_y(self, df_data):
    '''
     有研究表明，缩放标签有助于提升效果，因此进行验证
     至少它会帮助收敛
     https://stackoverflow.com/questions/36540745/pre-processing-data-normalizing-data-labels-in-regression
     https://stats.stackexchange.com/questions/111467/is-it-necessary-to-scale-the-target-value-in-addition-to-scaling-features-for-re
    '''
    # 特征进行标准化，标签进行归一化
    label_score = [116, 12, 8, 20, 8, 12, 16, 24,16]

    Compare = pd.DataFrame()
    normal_data = df_data.copy()
    x_data = normal_data.iloc[:, self.left:self.right]
    y_data = normal_data.iloc[:, -9:]
    X_scale = preprocessing.scale(x_data, axis=0)
    # y_scale = y_data/label_score
    y_mean = y_data.mean()
    y_std = y_data.std()
    # y_scale = (y_data - y_mean)/y_std
    y_scale = y_data - y_mean
    X_temp = pd.DataFrame(X_scale, columns=x_data.columns)
    Y_temp = pd.DataFrame(y_scale, columns=y_data.columns)
    # 合并name Person TEXT Frenchay
    Compare_temp = pd.concat([normal_data[['Person']], X_temp, Y_temp], axis=1)
    Compare = pd.concat([Compare, Compare_temp], axis=0)

    y_scaler = {}
    for label, mean, std in zip(y_data.columns, y_mean, y_std):
      y_scaler[label] = {'mean':mean, 'std':std}
    return Compare, y_scaler

  def fill_non(self, df_data):
    new_df = pd.DataFrame()
    for person in df_data['Person'].unique():
      person_data = df_data.loc[df_data['Person'] == person].reset_index(drop=True)
      new_df = pd.concat([new_df, person_data.fillna(person_data.mean())])
    return new_df.reset_index(drop=True)
      
  