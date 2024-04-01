#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 file_process.py
* @Time 	:	 2022/12/13
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import os
import pandas as pd
import csv

def csv_merge(target, *sources):
  '''
   @func csv_merge
   @desc merge csvs
   @usage:
    target = 'tmp/pitch_ac-acousticfeat_1208_kmeans.csv'
    csv_merge(target, 'tmp/pitch_ac-acousticfeat_1208_kmeans_Control.csv','tmp/pitch_ac-acousticfeat_1208_kmeans_Patient.csv')
   @param {}  
   @return {} 
  '''
  pd_list = []
  for sour_path in sources:
    pd_data = pd.read_csv(sour_path)
    pd_list.append(pd_data)
  source_data = pd.concat(pd_list, ignore_index=True)
  source_data.to_csv(target)

def add_task(label_csv, source_csv, target_csv):
  '''
   @func 
   @desc 
   @usage
      label_csv = ""
      source_csv = ""
      target_csv = ""
   @param {}  
   @return {} 
  '''
  # 'data/Label.csv'
  with open(label_csv) as lfp:
    person_label = {}
    reader = csv.DictReader(lfp)
    for row in reader:
      Frenchay = int(row['Frenchay'])
      if Frenchay == 116: # 正常病人
        degree = 0
      elif Frenchay < 50: # 中度
        degree = 2
      else:
        degree = 1
      label_dict = {'Frenchay':row['Frenchay'],'fanshe':row['fanshe'], 'huxi':row['huxi'], 'chun':row['chun'], 'he':row['he'], 'ruane':row['ruane'], 'hou':row['hou'], 'she':row['she'], 'yanyu':row['yanyu']}
      person_label[row['Person']] = label_dict
  
  base_feats_list = []
  with open(source_csv) as bfp:
    reader = csv.DictReader(bfp)
    for row in reader:
      # filename = row['\ufeffFilename']
      label_dict = person_label[row['Person']]
      row.update(label_dict)
      base_feats_list.append(row)
  df = pd.DataFrame(base_feats_list)
  df = df.drop(['\ufeffFilename'], axis=1)
  df.to_csv(target_csv)

def mkdir(targetdir):
  if not os.path.exists(targetdir):
    os.makedirs(targetdir)


# if __name__ == "__main__":
  # label_csv = "../data/Label.csv"
  # source_csv = "../data/result_intermediate/MonoAll_feats_Combined_0927.csv"
  # target_csv = "../data/result_intermediate/lsjori_acoustic_feats.csv"
  # add_task(label_csv, source_csv, target_csv)

  # work_path = '/mnt/shareEEx/liuxiaokang/data/MSDM/labeled_data/20230219/'
  # target = work_path + '20230219.csv'
  # csv_merge(target, work_path + 'N_10002_F_info.csv', work_path + 'N_10004_M_info.csv', work_path + 'N_10006_F_info.csv', work_path + 'N_10009_M_info.csv')
