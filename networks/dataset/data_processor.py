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
import numpy as np
import json
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from networks.dataset.video_process import calculate_optical_flow, read_video, normalize_video


def make_fbank(wav_file, 
               num_mel_bins=80, 
               frame_length=25, 
               frame_shift=10, 
               dither=0.0):
    waveform, sample_rate = torchaudio.load(wav_file)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    energy_floor=0.0,
                    sample_frequency=sample_rate)
    return mat


def make_mfcc(wav_file,
              num_mel_bins=23,
              frame_length=25,
              frame_shift=10,
              dither=0.0,
              num_ceps=12,
              high_freq=0.0,
              low_freq=20.0):
    waveform, sample_rate = torchaudio.load(wav_file)
    mfcc = T.MFCC(sample_rate=sample_rate,
                  n_mfcc=num_ceps,
                  melkwargs={'n_fft': 400, 'n_mels': num_mel_bins, 'hop_length': 160, 'win_length': 400})(waveform)
    delta = T.ComputeDeltas()(mfcc)
    delta_delta = T.ComputeDeltas()(delta)
    mfcc_dd = torch.cat([mfcc, delta, delta_delta], dim=1)
    return mfcc_dd.squeeze(0).transpose(0, 1)


def get_melspec(wav_file):
  waveform, sample_rate = torchaudio.load(wav_file)
  waveform = waveform.squeeze()
  mel_specgram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=1024, hop_length=512, win_length=400, center=True, pad_mode="reflect", power=2.0, norm="slaney", mel_scale="htk")(waveform)
  mel_specgram = mel_specgram.squeeze().transpose(0, 1)
  return mel_specgram


def get_rawwav(wav_file):
  speech_array, sampling_rate = torchaudio.load(wav_file)
  speech_array = speech_array.squeeze(0)
  return speech_array


def get_wav2vec(wav2vec_path):
  wav2vec_data = torch.load(wav2vec_path, map_location=torch.device('cpu'))
  wav2vec_data = torch.tensor(wav2vec_data, dtype=torch.float32)
  return wav2vec_data


def get_ahubert(hubert_path):
  file_name = os.path.basename(hubert_path).split('.')[0] + '.npy'
  hubert_data = np.load(os.path.join('data/MSDM/ahubert/', file_name))
  hubert_data = torch.from_numpy(hubert_data)
  return hubert_data


def get_encoder(encoder_path):
  encoder_data = torch.load(encoder_path)
  encoder_data = encoder_data.squeeze(0)
  return encoder_data


def get_crossatt(crossatt_path):
  crossatt_data = torch.load(crossatt_path)
  return crossatt_data[:, :-1]


def get_decoder_dist(decoder_dist_path):
  decoder_dist = torch.load(decoder_dist_path)
  decoder_dist = decoder_dist.squeeze(0)
  return decoder_dist[:, :-1]


def get_videorgb(video_path):
  # 判断文件后缀是不是avi
  if video_path.split('.')[-1] == 'avi':
    video_data = read_video(video_path)
  elif video_path.split('.')[-1] == 'pt':
    try:
      video_data = torch.load(video_path)
    except:
      print(video_path)
  video_data = normalize_video(video_data)
  return video_data


def get_cmlrv(video_path):
  file_name = video_path.split('.')[0] + '.cmlrv.pt'
  tensor_data = torch.load(file_name, map_location=torch.device('cpu'))
  # mean: tensor(-0.0116) std: tensor(0.5150)
  return tensor_data


def get_vhubert(video_path):
  file_name = os.path.basename(video_path).split('.')[0] + '.npy'
  hubert_data = np.load(os.path.join('data/MSDM/vhubert/', file_name))
  tensor_data = torch.from_numpy(hubert_data)
  return tensor_data


def get_videoflow(video_path):
  frames = np.load(video_path)
  flow_data = calculate_optical_flow(frames)
  tensor_data = torch.from_numpy(flow_data)
  return tensor_data


def get_gopfeat(gop_path):
  gop_data = torch.load(gop_path)
  return gop_data

def get_textemb(textemb_path):
  textemb_data = torch.load(textemb_path)
  textemb_data = textemb_data.squeeze(0)
  return textemb_data


def audio_videl_feat(data, audio_feat=None, video_feat=None, text_feat=None):
    for sample in data:
      obj = sample['src'] if 'src' in sample else sample
      # obj = json.loads(json_line)
      assert 'ID' in obj
      assert 'path' in obj
      assert 'label' in obj
      key = obj['ID']
      file_path = obj['path']
      label = obj['label']

      if audio_feat is not None:
        if audio_feat == 'wav':
          audio_data = get_rawwav(file_path + '.wav')
        elif audio_feat == 'fbank':
          audio_data = make_fbank(file_path + '.wav', num_mel_bins=80)
        elif audio_feat == 'mfcc':
          audio_data = make_mfcc(file_path + '.wav', num_mel_bins=15)
        elif audio_feat == 'melspec':
          audio_data = get_melspec(file_path + '.wav')
        elif audio_feat == 'wav2vec':
          audio_data = get_wav2vec(file_path + '.wav2vec.pt')
        elif audio_feat == 'ahubert':
          audio_data = get_ahubert(file_path)
        elif audio_feat == 'egemaps':
          audio_data = torch.load(file_path + '.egemaps.pt', map_location=torch.device('cpu'))
        elif audio_feat == 'encoder':
          audio_data = get_encoder(file_path + '.pt')
        elif audio_feat == 'crossatt':
          audio_data = get_crossatt(file_path + '.pt')
        elif audio_feat == 'decoder_dist':
          audio_data = get_decoder_dist(file_path + '.pt')
        elif audio_feat == 'cqcc':
          audio_data = torch.load(file_path + '.cqcc.pt')
        elif audio_feat == 'ivector':
          audio_data = torch.load(file_path + '.ivector.pt', map_location=torch.device('cpu')).to(torch.float32)
        else:
          raise ValueError('audio_feat not supported')
      else:
        audio_data = torch.zeros([10,80])

      if video_feat is not None:
        if video_feat == 'cropvideo_rgb':
          video_data = get_videorgb(file_path + '.pt')
        elif video_feat == 'flow':
          video_data = get_videoflow(file_path)
        elif video_feat == 'vhubert':
          video_data = get_vhubert(file_path)
        elif video_feat == 'cmlrv':
          video_data = get_cmlrv(file_path)
        else:
          raise ValueError('video_feat not supported')
      else:
        video_data = torch.zeros([3, 2, 96, 96]) # C, T, H, W
      
      if text_feat is not None:
        if text_feat == 'gop':
          gop_path = file_path.split('.')[0] + '.gop.pt'
          text_data = get_gopfeat(gop_path)
        elif text_feat == 'textemb':
          text_data = get_textemb(file_path + '.pt')
        else:
          raise ValueError('text_feat not supported')
      else:
        text_data = torch.zeros([2, 114])

      example = dict(key=key,
                      label=label,
                      audio_feat=audio_data,
                      video_feat=video_data,
                      text_feat=text_data)
      yield example


def cmvn(data, mean, istd):
  for sample in data:
    audio_data = sample['audio_feat']
    audio_data = (audio_data - mean) / istd
    sample['audio_feat'] = audio_data
    yield sample


def batch(data, batch_size=16):
  buf = []
  for sample in data:
    buf.append(sample)
    if len(buf) >= batch_size:
      yield buf
      buf = []
  if len(buf) > 0:
      yield buf


def video_padding(video_data, num_frames=0):
    maxlen = max([x.size(0) for x in video_data])
    set_len = num_frames if num_frames > 0 else maxlen
    padded_sequences = []
    for seq in video_data:
        data_dim = seq.dim()
        if data_dim == 4:
          T, C, H, W = seq.shape
        elif data_dim == 3:
          C, T, H, W = seq.shape
        elif data_dim == 2:
          T, C = seq.shape
        if T >= set_len:
            if data_dim == 4 or data_dim == 2:
              padded_sequences.append(seq[:set_len])
            elif data_dim == 3:
              padded_sequences.append(seq[:, :set_len])
        else:
            num_frames_to_pad = set_len - T
            if data_dim == 4:
              padded_seq = torch.cat([seq, torch.zeros(num_frames_to_pad, C, H, W)], dim=0)
            elif data_dim == 3:
              padded_seq = torch.cat([seq, torch.zeros(C, num_frames_to_pad, H, W)], dim=1)
            elif data_dim == 2:
              padded_seq = torch.cat([seq, torch.zeros(num_frames_to_pad, C)], dim=0)
            padded_sequences.append(padded_seq)
    return torch.stack(padded_sequences)


def audio_padding(audio_data, num_frames, padding_value):
    padded_sequences = []
    for seq in audio_data:
        if seq.dim() == 1: # wav
          target_len = num_frames*160 # 10ms
          if seq.size(0) >= target_len:
            padded_sequences.append(seq[:target_len])
          else:
            if padding_value == 'repeat':
              repeat_factor = target_len // seq.size(0) + 1
              data = seq.repeat((repeat_factor,))
              padded_sequences.append(data[:target_len])
            elif padding_value == 0:
              padded_seq = torch.cat([seq, torch.zeros(target_len - seq.size(0))], dim=0)
              padded_sequences.append(padded_seq)
        elif seq.dim() == 2: # fbank 或者 mfcc
          data_len, C = seq.shape
          if data_len >= num_frames: # chunk
              padded_sequences.append(seq[:num_frames])
          else: # padding
            if padding_value == 'repeat':
              repeat_factor = num_frames // data_len + 1
              data = seq.repeat((repeat_factor, 1))
              padded_sequences.append(data[:num_frames])
            elif padding_value == 0:
              padded_seq = torch.cat([seq, torch.zeros(num_frames - data_len, C)], dim=0)
              padded_sequences.append(padded_seq)
    return torch.stack(padded_sequences)


def audio_split_padding(audio_data, num_frames, num_split, padding_value):
    target_len = num_frames * num_split
    padded_data = audio_padding(audio_data, target_len, padding_value)
    return padded_data


def text_padding(text_data, num_token):
  padded_sequences = []
  for seq in text_data:
    data_len, C = seq.shape
    if data_len >= num_token:
      padded_sequences.append(seq[:num_token])
    else:
      repeat_factor = num_token // data_len + 1
      data = seq.repeat((repeat_factor, 1))
      padded_sequences.append(data[:num_token])
      # padded_seq = torch.cat([seq, torch.zeros(num_token - data_len, C)], dim=0)
      # padded_sequences.append(padded_seq)
  return torch.stack(padded_sequences)


def padding(data, a_framnum=0, v_framnum=0, num_token=0, audiosegnum=0):
    for sample in data:
        assert isinstance(sample, list)
        feats_length = torch.tensor([x['audio_feat'].size(0) for x in sample], dtype=torch.int32)
        order_audio = torch.argsort(feats_length, descending=True)
        feats_lengths = torch.tensor( [sample[i]['audio_feat'].size(0) for i in order_audio], dtype=torch.int32)
        sorted_feats = [sample[i]['audio_feat'] for i in order_audio]
        sorted_keys = [sample[i]['key'] for i in order_audio]
        sorted_labels = [ torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order_audio ]
        sorted_texts = [sample[i]['text_feat'] for i in order_audio]
        text_lengths = torch.tensor( [sample[i]['text_feat'].size(0) for i in order_audio], dtype=torch.int32)
        sorted_videos = [sample[i]['video_feat'] for i in order_audio]
        videos_lengths = torch.tensor( [sample[i]['video_feat'].size(0) for i in order_audio], dtype=torch.int32)

        # add a dim
        sorted_labels = [x.unsqueeze(0) for x in sorted_labels]
        if v_framnum == 0:
          padded_videos = pad_sequence(sorted_videos, batch_first=True, padding_value=0)
          text_lengths = videos_lengths # 注意，此处为权宜。因为text_lengths在这里没有用到，所以用videos_lengths代替
        else:
          padded_videos = video_padding(sorted_videos, v_framnum) # avcnn18, renet61

        if a_framnum == 0: # 使用fbank transformer/lstm时候使用这个
          padded_feats = pad_sequence(sorted_feats, batch_first=True, padding_value=0)
        else: # 使用dnn 时候使用repeat填充，dnn使用0填充
          if audiosegnum > 0:
            padded_feats = audio_split_padding(sorted_feats, a_framnum, audiosegnum, padding_value=0) # repeat, 0
          else:
            padded_feats = audio_padding(sorted_feats, a_framnum, padding_value='repeat')
        
        if num_token == 0:
          try:
            padded_text = pad_sequence(sorted_texts, batch_first=True, padding_value=0)
          except:
            print(sorted_keys)
            for i in range(len(sorted_texts)):
              print(sorted_texts[i].size())
            padded_text = text_padding(sorted_texts, num_token=5)
        else:
          padded_text = text_padding(sorted_texts, num_token=5)

        padded_labels = pad_sequence(sorted_labels, batch_first=True, padding_value=-1)

        yield (sorted_keys, padded_feats, padded_videos, padded_labels, feats_lengths, videos_lengths)


def test_data_processor():
  import pandas as pd
  csv_path = 'data/filtered_test.csv'
  raw_acu_feats = pd.read_csv(csv_path)
  row_dict_list = raw_acu_feats.T.to_dict().values()
  label = "classification"
  # generator = get_vhubert(row_dict_list, label)
  # generator = get_cmlrv(row_dict_list, label)
  generator = audio_videl_feat(row_dict_list, audio_feat='melspec')
  batch_generator = batch(generator, 8)
  pad_generator = padding(data=batch_generator, a_framnum=200)
  for i, example in enumerate(pad_generator):
    print(i)
    key, audio_feat, video_feat, text_feat, label, audio_len, text_len = example
    print(audio_feat.size())
    print(video_feat.size())
    print(text_feat.size())
    print(label)
    print(audio_len)
    print(text_len)
    if i == 10:
      break 
  print('done')


if __name__ == "__main__":
  test_data_processor()