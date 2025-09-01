#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 hubert_wav2vec_feat.py
* @Time 	:	 2023/02/13
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import pandas as pd
import numpy as np
import os, csv, sys
import torch
import json
# import torchaudio
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils
# from tools.multi_process import MultiProcess
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',30)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfSupervision():
  def __init__(self, file_path, result_path, feat_type, device='cuda:0') -> None:
    self.model_flag = ''
    self.mfcc = None
    self.model = None
    self.cfg = None
    self.file_set = set()
    self.file_path = file_path
    self.result_path = result_path
    self.wav_info_list = self.get_wav_info(file_path)
    self.device = torch.device(device)
    self.feat_type = feat_type

  def get_wav_info(self, file_path):
    wav_info_list = []
    if os.path.isdir(file_path):
      for root, dirs, files in os.walk(file_path, followlinks=True):
        for file in files:
          file_suff = file.split('.', 1)[-1]
          if file_suff == 'wav':
            wav_info = {'Path':os.path.join(root, file), 'Start':0, 'End':0}
            wav_info_list.append(wav_info)
    elif os.path.isfile(file_path):
      if file_path.endswith('csv'):
        df = pd.read_csv(file_path)
        list_of_dict = df.to_dict('records')
        for row in list_of_dict:
          tmp_dict = {}
          wav_name = os.path.basename(row['Path']).split('.')[0]
          time =  str(round(float(row['Start']),3)) + '_' + str(round(float(row['End']),3))
          tmp_dict['Segname'] = wav_name + '_' + time
          tmp_dict.update(row)
          wav_info_list.append(tmp_dict)
      elif file_path.endswith('json'):
        with open(file_path) as json_file:
          data = json.load(json_file)
        for id, path in data.items():
          wav_path = path.split('.')[0] + '.wav'
          wav_info_list.append({'Path':wav_path, 'Start':0, 'End':0})
    else:
      exit(-1)
    return wav_info_list

  def _postprocess(self, feats, normalize=False):
    if feats.dim() == 2:
      feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
      with torch.no_grad():
        feats = F.layer_norm(feats, feats.shape)
    return feats
  
  def _load_model(self, model_path):
    print("loading model(s) from {}".format(model_path))
    models, self.cfg, task = checkpoint_utils.load_model_ensemble_and_task(
      [model_path],
      suffix="",
    )
    print("loaded model(s) from {}".format(model_path))
    print(f"normalize: {self.cfg.task.normalize}")
    self.model = models[0]
    self.model = self.model.to(self.device)
    self.model = self.model.half()
    self.model.eval()

  def _get_input(self, wav_info):
    wav_path = wav_info['Path']
    wav, sr = sf.read(wav_path)
    if wav_info['Start'] == wav_info['End']:
      start_frame = 0
      end_frame = -1
    else:
      start_frame = int(float(wav_info['Start']) * sr)
      end_frame = int(float(wav_info['End']) * sr)
    feat = torch.from_numpy(wav[start_frame:end_frame]).float()
    feat = self._postprocess(feat, normalize=self.cfg.task.normalize)
    feats = feat.view(1, -1)
    padding_mask = (
    torch.BoolTensor(feats.shape).fill_(False)
    )
    inputs = {
        "source": feats.half().to(self.device),
        "padding_mask": padding_mask.to(self.device),
    }
    return inputs

  def _extract_wav2vec(self, wav_info, save_path):
    # save_path = self.result_path + 'data/' + wav_info['Segname']
    # save_path = wav_info['Path'].split('.')[0]
    file_name = os.path.basename(wav_info['Path']).split('.')[0]
    save_path += '/' + file_name
    feat_path = save_path + '.wav2vec.pt'
    mean_feat_path = save_path + '.w2v_mean.pt'
    max_feat_path = save_path + '.w2v_max.pt'

    if not os.path.exists(feat_path) or not os.path.exists(mean_feat_path) or not os.path.exists(max_feat_path):
      inputs = self._get_input(wav_info)
      with torch.no_grad():
        logits = self.model.extract_features(**inputs)
      # feat = logits[0].detach().cpu().numpy()[0]
      # feat = logits[0] # hubert
      feat = logits['x'] # wav2vec
      feat = torch.squeeze(feat)
      mean_feat = torch.mean(feat, dim=0)
      max_feat = torch.max(feat, dim=0).values
      torch.save(feat, feat_path)
      torch.save(mean_feat, mean_feat_path)
      torch.save(max_feat, max_feat_path)
  
  def _extract_hubert(self, wav_info, save_path):
    feat_path = save_path + '.hubert.pt'
    mean_feat_path = save_path + '.hub_mean.pt'
    max_feat_path = save_path + '.hub_max.pt'

    if not os.path.exists(feat_path) or not os.path.exists(mean_feat_path) or not os.path.exists(max_feat_path):
      inputs = self._get_input(wav_info)
      with torch.no_grad():
        logits = self.model.extract_features(**inputs)
      feat = logits[0].detach().cpu().numpy()[0]
      feat = logits[0] # hubert
      feat = torch.squeeze(feat)
      mean_feat = torch.mean(feat, dim=0)
      max_feat = torch.max(feat, dim=0).values
      torch.save(feat, feat_path)
      torch.save(mean_feat, mean_feat_path)
      torch.save(max_feat, max_feat_path)

  def _extract_output(self, wav_info):
    if os.path.isdir(self.file_path):
      save_path = wav_info['Path'].split('.')[0]
    elif os.path.isfile(self.file_path):
      if self.file_path.endswith('.csv'):
        save_path = self.result_path + 'data/' + wav_info['Segname']
      elif self.file_path.endswith('.json'):
        save_path = self.result_path

    if self.feat_type == "wav2vec":
      self._extract_wav2vec(wav_info, save_path)
    elif self.feat_type == "hubert":
      try:
        self._extract_hubert(wav_info, save_path)
      except:
        print(save_path, 'error')
  
  def interface_hidden_feat(self, model_path, multi_num=1):
    self._load_model(model_path)
    if multi_num != 1:
      # MP = MultiProcess()
      # MP.multi_not_result(func=self._extract_output, arg_list=self.wav_info_list, process_num=multi_num)
      print('multi process not support')
    else:
      for wav_path in self.wav_info_list:
        self._extract_output(wav_path)
    
    if self.file_path.endswith('csv'): # segment级别csv需要保存
      df = pd.DataFrame(self.wav_info_list)
      new_order = ['Person', 'Path', 'Segname', 'Start', 'End', 'Frenchay', 'fanshe', 'huxi', 'chun', 'he', 'ruane', 'hou', 'she', 'yanyu']
      df = df[new_order]
      df.to_csv(self.file_path.split('.')[0] + '_phasesegment.csv',index=False)
      print(len(self.file_set))

def read_feats(dir):
  shape_list = []
  for root, dirs, files in os.walk(dir, followlinks=True):
    for file in files:
      if file.endswith('.wav2vec.pt'):
        feat = torch.load(os.path.join(root, file))
        print(feat.shape)
        shape_list.append(feat.shape[0])
  print(np.mean(shape_list), np.max(shape_list), np.min(shape_list))
  hist, bins = np.histogram(shape_list, bins=15)
  print(hist, bins)

if __name__ == "__main__":
  wav_dir = '/tmp/MSDM/crop_video/cropvideo.json'
  device = 'cuda:0'
  feat_type = 'wav2vec'
  if feat_type == 'wav2vec':
    model_path = "exp/wav2vec_hubert/wav2vec2-large/chinese-wav2vec2-large.pt"
    # model_path = "exp/wav2vec_hubert/wav2vec2-base/chinese-wav2vec2-base.pt"
    # model_path = "exp/wav2vec_hubert/XLSR-53/xlsr_53_56k.pt"
  elif feat_type == 'hubert':
    model_path = "exp/wav2vec_hubert/hubert-base/chinese-hubert-base.pt"

  result_path = 'data/MSDM/wav2vec_large'
  # SP = SelfSupervision(wav_dir, result_path, feat_type, device)
  # SP.interface_hidden_feat(model_path, 1)
  read_feats(result_path)

  print("finished")



