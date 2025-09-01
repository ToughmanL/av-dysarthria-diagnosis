#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 extract_eGeMAPs.py
* @Time 	:	 2023/03/14
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 compute egemaps
'''

import os
import csv
import numpy as np
import torch
import pandas as pd

opensmile_path = 'tool/opensmile-3.0-linux-x64/'
egemaps_conf = opensmile_path + 'config/egemaps/v01b/eGeMAPSv01b.conf'
# ComParE_conf = opensmile_path + 'config/compare16/ComParE_2016.conf'
pathExcuteFile = opensmile_path + 'bin/SMILExtract'

class ComputeEgemaps():
  def __init__(self, data_dir, suffix='.wav'):
    self.wav_path_list = []
    self._get_wavs(data_dir, suffix)

  def _get_wavs(self, data_dir, suffix):
    for root, dirs, files in os.walk(data_dir):
      for file in files:
        # suffix check
        if file.endswith(suffix):
          wav_path = os.path.join(root, file)
          self.wav_path_list.append(wav_path)
  
  def _egemaps_command(self, wav_path, egemaps_path):
    if os.path.exists(egemaps_path):
      os.remove(egemaps_path)
    cmd = '{exec} -C {config} -I {wav} -csvoutput {egemaps}'.format(exec=pathExcuteFile, config=egemaps_conf, wav=wav_path, egemaps=egemaps_path)
    print(cmd)
    os.system(cmd)

  def _read_egemaps(self, egemaps_path):
    egemaps_data = pd.read_csv(egemaps_path, sep=';')
    num_data = np.array(egemaps_data)
    return num_data[0][2:]

  def _write_pt(self, egemaps_data, pt_path):
    egemaps_data = torch.tensor(egemaps_data)
    torch.save(egemaps_data, pt_path)

  def get_egmaps(self, feat_dir):
    for wav_path in self.wav_path_list:
        file_name = os.path.basename(wav_path)
        csv_path = os.path.join(feat_dir, file_name.replace('.wav', '.egemaps.csv'))
        pt_path = os.path.join(feat_dir, file_name.replace('.wav', '.egemaps.pt'))
        self._egemaps_command(wav_path, csv_path)
        egemaps_data = self._read_egemaps(csv_path)
        self._write_pt(egemaps_data, pt_path)
  
  def compute_menstd(self, feat_dir):
    all_data = []
    for egemaps_path in self.wav_path_list:
      egemaps_data = torch.load(egemaps_path)
      all_data.append(egemaps_data)
    all_data = torch.stack(all_data)
    mean = all_data.mean(dim=0)
    std = all_data.std(dim=0)
    return mean, std
  
  def normalize(self, feat_dir, mean, std):
    for egemaps_path in self.wav_path_list:
      egemaps_data = torch.load(egemaps_path)
      egemaps_data = (egemaps_data - mean) / std
      pt_path = os.path.join(feat_dir, os.path.basename(egemaps_path))
      self._write_pt(egemaps_data, pt_path)



if __name__ == "__main__":
#   segment_dir = 'data/MSDM/crop_audio/seg_data/'
  segment_dir = 'data/MSDM/crop_audio/seg_data/'
  feat_dir = 'data/MSDM/egemaps/'
  new_dir = 'data/MSDM/egemapsnorm/'
  CE = ComputeEgemaps(feat_dir, suffix='.egemaps.pt')
  # CE.get_egmaps(feat_dir=feat_dir)
  mean, std = CE.compute_menstd(feat_dir)
  CE.normalize(new_dir, mean, std)