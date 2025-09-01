#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 evaluation.py
* @Time 	:	 2023/12/25
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import numpy as np
import pandas as pd
from sklearn import metrics

def regression_eval(label, predict):
  syllabel_test_r2 = metrics.r2_score(label, predict)
  syllabel_test_rmse = np.sqrt(metrics.mean_squared_error(label, predict))
  result_info = {'rmse':syllabel_test_rmse, 'r2':syllabel_test_r2}
  return result_info

def classification_eval(label, predict):
  label = pd.to_numeric(label)
  labels = label.drop_duplicates().sort_values().tolist()
  acc = metrics.accuracy_score(label, predict)
  confusion_matrix = metrics.confusion_matrix(label, predict, labels=labels)

   # 计算每一类的精确度、召回率和 F1 值
  precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(label, predict, labels=labels)
  result_info = {'acc':acc}

  # 计算总精确度、召回率和 F1 值
  total_support = sum(metrics.precision_recall_fscore_support(label, predict, labels=labels)[3])
  total_precision = sum([precision[i] * metrics.precision_recall_fscore_support(label, predict, labels=labels)[3][i] for i in range(len(labels))]) / total_support
  # 计算总召回率
  total_recall = sum([recall[i] * metrics.precision_recall_fscore_support(label, predict, labels=labels)[3][i] for i in range(len(labels))]) / total_support
  # 计算总 F1 值
  total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
  result_info['total_precision'] = total_precision
  result_info['total_recall'] = total_recall
  result_info['total_f1'] = total_f1

  # 混淆矩阵
  result_info['confusion_matrix'] = confusion_matrix
  return result_info

def classification_eval_person(result_df, label_name):
  all_data_counts = result_df[label_name].value_counts(ascending=True).to_dict()
  all_data_counts = dict(sorted(all_data_counts.items()))
  grouped = result_df.groupby('Person')
  person_acc = {'all_data_counts':all_data_counts}
  for name, group in grouped:
    acc = metrics.accuracy_score(group[label_name], group['Predict'])
    predict_counts = group['Predict'].value_counts(ascending=True).to_dict()
    predict_counts = dict(sorted(predict_counts.items()))
    label = group[label_name].iloc[0]
    num_utt = len(group)
    person_acc[name] = {'ACC':"{:.4f}".format(acc), 'label': label, 'num_utt':num_utt, 'predict_counts':predict_counts}
  return person_acc


def result_process(result_tuple):
  key_list, hpy_list, ref_list = result_tuple
  key_array = np.array(key_list)
  ref_array = np.array(ref_list).flatten()
  ref_array = pd.Series(ref_array)
  X_test_predict = np.array(hpy_list)
  return ref_array, X_test_predict