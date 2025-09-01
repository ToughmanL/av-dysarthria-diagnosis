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


def mkdir(targetdir):
  if not os.path.exists(targetdir):
    
    os.makedirs(targetdir)

def id2person(id):
  item = id.replace('repeat_', '')
  nf_num = item.split('_')[0:3]
  person = nf_num[0] + '_' + nf_num[2] + '_' + nf_num[1]
  return person

# if __name__ == "__main__":
  # label_csv = "../data/Label.csv"
  # source_csv = "../data/result_intermediate/MonoAll_feats_Combined_0927.csv"
  # target_csv = "../data/result_intermediate/lsjori_acoustic_feats.csv"
  # add_task(label_csv, source_csv, target_csv)

  # work_path = 'data/MSDM/labeled_data/20230219/'
  # target = work_path + '20230219.csv'
  # csv_merge(target, work_path + 'N_10002_F_info.csv', work_path + 'N_10004_M_info.csv', work_path + 'N_10006_F_info.csv', work_path + 'N_10009_M_info.csv')
