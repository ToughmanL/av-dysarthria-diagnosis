#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	av_crossmodel.py
 *@Time	: 2023-11-04 16:36:04
 *@Author	:	lxk
 *@Version	:	1.0
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''
import torch
from torch import nn
import torch.nn.functional as F

class AudioCNN(torch.nn.Module):
  def __init__(self, audio_dim=64, out_dim=2):
    super().__init__()
    self.aconv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.aconv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.aconv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1,padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.audio_fc = nn.Linear(960, audio_dim) # mfcc:384, fbank:960
    self.fc = nn.Linear(audio_dim, out_dim)


  def forward(
      self, 
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor
  ):
    x = F.relu(self.aconv1(speech.unsqueeze(1)))
    x = self.pool(x)
    x = F.relu(self.aconv2(x))
    x = self.pool(x)
    x = F.relu(self.aconv3(x))
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.audio_fc(x))
    x = self.fc(x)
    return x


class Video3DCNN(torch.nn.Module):
  def __init__(self, video_dim=64, out_dim=2):
    super().__init__()
    self.vconv1 = nn.Conv3d(in_channels=3, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.vconv2 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.vconv3 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1,padding=1)
    self.pool3d = nn.MaxPool3d(2, 2)
    self.video_fc = nn.Linear(9216, video_dim) # 18:9216, 61:32256
    self.fc = nn.Linear(video_dim, out_dim)


  def forward(self, 
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor
  ):
    x = video.permute(0, 2, 1, 3, 4)
    x = F.relu(self.vconv1(x))
    x = self.pool3d(x)
    x = F.relu(self.vconv2(x))
    x = self.pool3d(x)
    x = F.relu(self.vconv3(x))
    x = self.pool3d(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.video_fc(x))
    x = self.fc(x)
    return x

class VideoCNN(torch.nn.Module):
  def __init__(self, video_dim=64, out_dim=2):
    super().__init__()

    self.pool3d = nn.MaxPool3d(2, 2)
    self.video_fc = nn.Linear(9216, video_dim) # 18:9216, 61:32256
    self.fc = nn.Linear(video_dim, out_dim)

  def threeD_to_2D_tensor(self, x):
      n_batch, n_channels, s_time, sx, sy = x.shape
      x = x.transpose(1, 2).contiguous()
      return x.reshape(n_batch*s_time, n_channels, sx, sy)

  def forward(self, 
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor
  ):
    x = video.permute(0, 2, 1, 3, 4)
    x = F.relu(self.vconv1(x))
    x = self.pool3d(x)
    x = F.relu(self.vconv2(x))
    x = self.pool3d(x)
    x = F.relu(self.vconv3(x))
    x = self.pool3d(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.video_fc(x))
    x = self.fc(x)
    return x


class AudioVideoCNN(torch.nn.Module):
  def __init__(self, audio_dim=64, video_dim=64, out_dim=2):
    super().__init__()
    self.aconv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.aconv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.aconv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1,padding=1)

    self.vconv1 = nn.Conv3d(in_channels=3, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.vconv2 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.vconv3 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1,padding=1)
    
    self.pool = nn.MaxPool2d(2, 2)
    self.pool3d = nn.MaxPool3d(2, 2)

    self.audio_fc = nn.Linear(480, audio_dim) # 30:480, 200:4000
    self.video_fc = nn.Linear(9216, video_dim) # 18:9216, 61:32256
    self.fc = nn.Linear(audio_dim + video_dim, out_dim)
    self.av_dim = audio_dim + video_dim

  def forward(
      self, 
      speech: torch.Tensor,
      video: torch.Tensor,
      speech_lens: torch.Tensor
  ):
    audio_x = F.relu(self.aconv1(speech.unsqueeze(1)))
    audio_x = self.pool(audio_x)
    audio_x = F.relu(self.aconv2(audio_x))
    audio_x = self.pool(audio_x)
    audio_x = F.relu(self.aconv3(audio_x))
    audio_x = self.pool(audio_x)
    audio_x = audio_x.view(audio_x.size(0), -1)
    audio_x = F.relu(self.audio_fc(audio_x))

    video = video.permute(0, 2, 1, 3, 4)
    video_x = F.relu(self.vconv1(video))
    video_x = self.pool3d(video_x)
    video_x = F.relu(self.vconv2(video_x))
    video_x = self.pool3d(video_x)
    video_x = F.relu(self.vconv3(video_x))
    video_x = self.pool3d(video_x)
    video_x = video_x.view(video_x.size(0), -1)
    video_x = F.relu(self.video_fc(video_x))

    av_out = torch.cat((audio_x, video_x), 1)
    av_out = self.fc(av_out)
    return av_out
  
  def output_size(self):
    return self.av_dim



if __name__ == '__main__':
  audio_data = torch.randn(64, 30, 45) # B, C, T, F
  video_data = torch.randn(64, 18, 3, 96, 96) # B, C, T, H, W
  AVC = AudioVideoCNN()
  res = AVC.forward(audio_data, video_data, None, None, None, None)
  print(res)