#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 data_processor.py
* @Time 	:	 2023/07/21
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import os
import logging
import json
import random
import torch
import torchaudio
import cv2
import numpy as np

from torchvision.io import read_video
import torchvision.transforms.functional as F
from networks.dataset.transforms import get_transform

class DataProcessor():
  def __init__(self, label, target_len, setment_dir=None) -> None:
    self.name_path_dict = {}
    self.label = label
    self.batch_size = 16
    self.target_len = target_len
    # self._get_name_path(setment_dir)
    self.setment_dir = setment_dir

  def _get_name_path(self, setment_dir):
    for root, dirs, files in os.walk(setment_dir, followlinks=True):
      for file in files:
        if len(file.split('.')) < 2:
          continue
        name, suff = file.split('.')[0], file.split('.')[1]
        if suff == 'wav':
          self.name_path_dict[name] = os.path.join(root, file)
  
  def _get_file_path(self, file_name, suffix):
    name_ll = file_name.split('_')
    if len(name_ll) == 7:
      pass
    elif len(name_ll) == 8:
      name_ll = name_ll[1:]
    else:
      print(file_name, 'error file')
      exit(-1) 
    N_S_name = "Control" if name_ll[0] == "N" else "Patient"
    person_name = name_ll[0] + '_' + name_ll[2] + '_' + name_ll[1]
    file_path = os.path.join(self.setment_dir, N_S_name, person_name, file_name + '.' + suffix)
    return file_path

  def _crop_lip(self, input_tensor, crop_size):
    h, w, d = input_tensor.shape
    crop_w, crop_d = crop_size

    start_w = (w - crop_w) // 2
    end_w = start_w + crop_w

    start_d = (d - crop_d) // 2
    end_d = start_d + crop_d

    cropped_tensor = input_tensor[:, start_w:end_w, start_d:end_d]
    return cropped_tensor

  def _read_and_convert_to_grayscale(self, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      # Convert to grayscale
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Convert to PyTorch tensor
      tensor_frame = F.to_tensor(gray_frame)
      # Normalize the tensor (optional)
      tensor_frame = F.normalize(tensor_frame, [0.5], [0.5])
      if tensor_frame.shape[2] > 80:
        tensor_frame = self._crop_lip(tensor_frame, (80, 80))
      frames.append(tensor_frame)
    cap.release()
    if len(frames) == 0:
      tensor_frame = torch.rand(1,80,80)
      frames.append(tensor_frame)
      frames.append(tensor_frame)
    return torch.stack(frames)

  def _sequence_pad(self, feat, target_len):
    if feat.shape[0] > target_len:
      feat = feat[:target_len,:]
    else:
      pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, target_len-feat.shape[0]))
      feat = pad(feat)
    return feat

  def compute_fbank(self, data,
              num_mel_bins=23,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_rate= 16000):
    # waveform = waveform * (1 << 15)
    random.seed(0)
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    for dataframe_dict in data:
      label_score = torch.as_tensor(dataframe_dict['src'][self.label])
      fbank_list = []
      for vowel in select_vowel_list:
        select_feat_path = dataframe_dict['src'][vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        # wav_path = self.name_path_dict[wav_name]
        wav_path = self._get_file_path(wav_name, 'wav')
        waveform, sr = torchaudio.load(wav_path)
        fbank = torchaudio.compliance.kaldi.fbank(waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    energy_floor=0.0,
                    sample_frequency=sample_rate)
        if fbank.shape[1] > self.target_len:
          fbank = fbank[:,:self.target_len]
        else:
          pad = torch.nn.ZeroPad2d(padding=(0, self.target_len-fbank.shape[1], 0, 0))
          fbank = pad(fbank)
        fbank_list.append(fbank)
      fbank_feat = torch.cat(fbank_list, 1)
      yield fbank_feat, label_score

  def compute_mfcc(self, data,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=13,
                 high_freq=0.0,
                 low_freq=20.0,
                 sample_rate=16000):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    target_len = 82
    for dataframe_dict in data:
      label_score = torch.as_tensor(dataframe_dict['src'][self.label])
      mfcc_list = []
      for vowel in select_vowel_list:
        select_feat_path = dataframe_dict['src'][vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        # wav_path = self.name_path_dict[wav_name]
        wav_path = self._get_file_path(wav_name, 'wav')
        waveform, sr = torchaudio.load(wav_path)
        # Only keep key, feat, label
        mfcc = torchaudio.compliance.kaldi.mfcc(waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    num_ceps=num_ceps,
                    high_freq=high_freq,
                    low_freq=low_freq,
                    sample_frequency=sample_rate)
        if mfcc.shape[0] > target_len:
          pad_mfcc = mfcc[:target_len,:]
        else:
          pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, target_len-mfcc.shape[0]))
          pad_mfcc = pad(mfcc)
        # first_diff_mfcc = torch.diff(pad_mfcc, n=1, dim=0)
        # second_diff_mfcc = torch.diff(first_diff_mfcc, n=1, dim=0)
        # concat_mfcc = torch.cat((pad_mfcc[:80,:], first_diff_mfcc[:80,:], second_diff_mfcc), dim=1)
        # mfcc_list.append(torch.transpose(concat_mfcc,0,1))
        mfcc_list.append(torch.transpose(pad_mfcc[:80,:],0,1))
      mfcc_feat = torch.stack(mfcc_list, 0)
      yield mfcc_feat, label_score.float()

  def compute_stft(self, data):
    loop = False
    if loop:
      random.seed(0)
      select_vowel_list = random.sample(['a', 'o', 'e', 'i', 'u', 'v'], 1)
      for dataframe_dict in data:
        label_score = torch.as_tensor(dataframe_dict['src'][self.label])
        select_feat_path = dataframe_dict['src'][select_vowel_list[0]+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        # wav_path = self.name_path_dict[wav_name]
        wav_path = self._get_file_path(wav_name, 'wav')
        waveform, sr = torchaudio.load(wav_path)
        stft_complex = torch.stft(waveform, 1139, hop_length=24, win_length=32, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        stft = stft_complex[:,:,:,0] # 取复数的实部
        target_len = 450
        if stft.shape[2] < target_len:
          pad = torch.nn.ZeroPad2d(padding=(0, target_len-stft.shape[2], 0, 0))
          stft = pad(stft)
        elif stft.shape[2] > target_len:
          stft = stft[:,:,:target_len]
        yield (stft.to(torch.float32), label_score.to(torch.float32))
    else:
      for dataframe_dict in data:
        obj = dataframe_dict['src']
        label_score = torch.as_tensor(obj[self.label])
        wav_path = obj['Path']
        waveform, sr = torchaudio.load(wav_path)
        stft_complex = torch.stft(waveform, 1139, hop_length=24, win_length=32, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        stft = stft_complex[:,:,:,0] # 取复数的实部
        target_len = 450 # 按照文章标准
        if stft.shape[2] < target_len:
          pad = torch.nn.ZeroPad2d(padding=(0, target_len-stft.shape[2], 0, 0))
          stft = pad(stft)
        elif stft.shape[2] > target_len:
          stft = stft[:,:,:target_len]
        yield (stft.to(torch.float32), label_score.to(torch.float32))
  
  def get_cmlrv(self, data, label):
    for dataframe_dict in data:
      if 'src' in dataframe_dict:
        obj = dataframe_dict['src']
      else:
        obj = dataframe_dict
      label_score = torch.as_tensor(obj[label])
      tensor_data = torch.load(obj['path'], map_location=torch.device('cpu'))
      yield tensor_data, label_score.float()
    # mean: tensor(-0.0116) std: tensor(0.5150)

  def get_wavsegment(self, data,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=13,
                 high_freq=0.0,
                 low_freq=20.0,
                 sample_rate=16000):
    target_len = 180 # 论文中提供
    for dataframe_dict in data:
      obj = dataframe_dict['src']
      label_score = torch.as_tensor(obj[self.label])
      wav_path = obj['Path']
      sample_rate = torchaudio.backend.sox_io_backend.info(wav_path).sample_rate
      start_frame = int(float(obj['Start']) * sample_rate)
      end_frame = int(float(obj['End']) * sample_rate)
      waveform, _ = torchaudio.backend.sox_io_backend.load(
          filepath=wav_path,
          num_frames=end_frame - start_frame,
          frame_offset=start_frame)
      mfcc = torchaudio.compliance.kaldi.mfcc(waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    num_ceps=num_ceps,
                    high_freq=high_freq,
                    low_freq=low_freq,
                    sample_frequency=sample_rate)
      if mfcc.shape[0] > target_len:
        pad_mfcc = mfcc[:target_len,:]
      else:
        pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, target_len-mfcc.shape[0]))
        pad_mfcc = pad(mfcc)
      # first_diff_mfcc = torch.diff(pad_mfcc, n=1, dim=0)
      # second_diff_mfcc = torch.diff(first_diff_mfcc, n=1, dim=0)
      # concat_mfcc = torch.cat((pad_mfcc[:80,:], first_diff_mfcc[:80,:], second_diff_mfcc), dim=1)
      # mfcc_list.append(torch.transpose(concat_mfcc,0,1))
      mfcc_feat = torch.transpose(pad_mfcc[:target_len,:],0,1)
      yield mfcc_feat, label_score.float()

  def get_wav2vecmax(self, data):
    for dataframe_dict in data:
      obj = dataframe_dict['src']
      label_score = torch.as_tensor(obj[self.label])
      wav2vec_mean_path = 'data/segment_data/PhaseSegData/data/' + obj['Segname'] + '.w2v_max.pt'
      wav2vec_data = torch.load(wav2vec_mean_path, map_location=torch.device('cpu'))
      yield wav2vec_data.float(), label_score.float()
  
  def get_hubertmax(self, data):
    for dataframe_dict in data:
      obj = dataframe_dict['src']
      label_score = torch.as_tensor(obj[self.label])
      wav2vec_mean_path = 'data/segment_data/PhaseSegData/data/' + obj['Segname'] + '.hub_max.pt'
      wav2vec_data = torch.load(wav2vec_mean_path, map_location=torch.device('cpu'))
      yield wav2vec_data.float(), label_score.float()

  def get_vhubert(self, data, label):
    for dataframe_dict in data:
      if 'src' in dataframe_dict:
        obj = dataframe_dict['src']
      else:
        obj = dataframe_dict
      label_score = torch.as_tensor(obj[label])
      npy_path = np.load(obj['path'])
      tensor_data = torch.from_numpy(npy_path)
      yield tensor_data.float(), label_score.float()
  
  def get_videoflow(self, data, label):
    for dataframe_dict in data:
      if 'src' in dataframe_dict:
        obj = dataframe_dict['src']
      else:
        obj = dataframe_dict
      label_score = torch.as_tensor(obj[label])
      avi_path = np.load(obj['path'])
      cap = cv2.VideoCapture(avi_path)
      frames = []
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        frames.append(gray_frame)
      tensor_data = get_transform(frames)
      yield tensor_data.float(), label_score.float()

  def padding(self, data):
    for stft, label_score in data:
      if stft.shape[2] < self.target_len:
        pad = torch.nn.ZeroPad2d(padding=(0, self.target_len-stft.shape[2], 0, 0))
        stft = pad(stft)
      elif stft.shape[2] > self.target_len:
        stft = stft[:,:,:self.target_len]
      yield stft, label_score

  def batch(self, data):
    feat_buf, label_buf = [], []
    for stft, label_score in data:
      feat_buf.append(stft)
      label_buf.append(label_score)
      if len(feat_buf) >= self.batch_size:
        yield torch.stack(feat_buf), torch.stack(label_buf)
        feat_buf, label_buf = [], []
    if len(feat_buf) > 0:
      yield torch.stack(feat_buf), torch.stack(label_buf)

def AVDataBatch(samples):
  audio_modality, video_modality = False, False
  if samples[0]['loop_feat'] != None:
    audio_modality = True
  if samples[0]['video_vowels'] != None:
    video_modality = True
  batch_loop_feat, batch_avi_data, batch_label = [], [], []
  if video_modality:
    max_len = max(tensor.shape[0] for sublist in samples for tensor in sublist['video_vowels'])
  for sample in samples:
    if audio_modality:
      batch_loop_feat.append(sample['loop_feat'])
    else:
      batch_loop_feat.append(torch.zeros(20))
    if video_modality:
      six_vowels_list = []
      for video_tensor in sample['video_vowels']:
        zeros_to_pad = torch.zeros((max_len-video_tensor.shape[0], *tuple(video_tensor.size()[1:])))
        padded_tensor =  torch.cat((video_tensor, zeros_to_pad), dim=0)
        trans_video_tensor = torch.transpose(padded_tensor, 0, 1)
        six_vowels_list.append(trans_video_tensor.unsqueeze(0))
      six_vowels_tensor = torch.cat(six_vowels_list, dim=0)
      batch_avi_data.append(six_vowels_tensor)
    else:
      batch_avi_data.append(torch.zeros(6, 1, 1, 80, 80))
    batch_label.append(sample['label'])
  batch_loop_feat_tensor = torch.stack(batch_loop_feat)
  batch_avi_data_tensor = torch.stack(batch_avi_data)
  batch_label_tensor = torch.stack(batch_label)
  return batch_loop_feat_tensor, batch_avi_data_tensor, batch_label_tensor


def test_hubert():
  import pandas as pd
  csv_path = 'data/test.csv'
  raw_acu_feats = pd.read_csv(csv_path)
  row_dict_list = raw_acu_feats.T.to_dict().values()
  label = "classification"
  DP = DataProcessor(label=label, target_len=50)
  # generator = DP.get_vhubert(row_dict_list, label)
  generator = DP.get_cmlrv(row_dict_list, label)
  for data, label in generator:
    print(data.size())
    print(label)

if __name__ == "__main__":
  test_hubert()