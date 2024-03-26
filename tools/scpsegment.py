#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 scpsegment.py
* @Time 	:	 2024/03/08
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import os
import cv2
from multi_process import MultiProcess 
import warnings
warnings.filterwarnings('ignore')


class ScpSegment():
  def __init__(self, scp_dir, out_dir) -> None:
    self.scp_dir = scp_dir
    self.name_path_dict, self.name_file_time_dict = {}, {}
    self.out_dir = out_dir

  def read_scp(self):
    with open(self.scp_dir+'/wav.scp', 'r') as fp:
      for line in fp:
        name_path = line.rstrip('\n').split()
        self.name_path_dict[name_path[0]] = name_path[1] # filename:filepath

    with open(self.scp_dir+'/segments', 'r') as fp:
      for line in fp:
        nft = line.rstrip('\n').split()
        self.name_file_time_dict[nft[0]] = {'filename':nft[1], 'st':nft[2], 'ed':nft[3]} # segmentname: filename, starttime, endtime

  def seg_audio(self, segname):
    filetime = self.name_file_time_dict[segname]
    ori_wav_path = self.name_path_dict[filetime['filename']]
    output_wav_path = self.out_dir + '/' + segname + '.wav'
    if not os.path.exists(output_wav_path):
      conson_audio_command = 'sox ' + ori_wav_path + ' ' + output_wav_path + ' trim ' + str(filetime['st']) + ' ' + str(filetime['ed'])
      os.system(conson_audio_command)

  def seg_video(self, segname):
    filetime = self.name_file_time_dict[segname]
    ori_avi_path = self.name_path_dict[filetime['filename']].replace('.wav', '.avi')
    # 获取视频时间
    cap = cv2.VideoCapture(ori_avi_path)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))  # 帧速率
    frame_num = cap.get(7)
    video_duration = frame_num / video_fps
    video_start = round(float(filetime['st']), 3) - 1 # 前置1秒，其中静止0.4秒，AIT0.6秒
    if video_start < 0:
      print('video start less than 0 {}'.format(segname))
      return
    # video_start = 0 if video_start < 0 else video_start
    video_end = round(float(filetime['ed']), 3) + 0.3 # 后移0.3秒
    if video_end > video_duration:
      print('video end greater than duration {}'.format(segname))
      return
    # video_end = video_duration if video_end > video_duration else video_end

    output_avi_path = self.out_dir + '/' + segname + '.avi'
    if not os.path.exists(output_avi_path):
      conson_video_command = 'ffmpeg -loglevel quiet -y -i ' + ori_avi_path + ' -ss ' + str(video_start) + ' -to ' + str(video_end) + ' -c copy ' + output_avi_path
      os.system(conson_video_command)

  def seg_data(self, multi=1):
    self.read_scp()
    if multi == 1:
      for segname in self.name_file_time_dict.keys():
        self.seg_audio(segname)
        self.seg_video(segname)
    else:
      MP = MultiProcess()
      # MP.multi_not_result(func=self.seg_audio, arg_list=self.name_file_time_dict.keys())
      MP.multi_not_result(func=self.seg_video, arg_list=self.name_file_time_dict.keys())

if __name__ == "__main__":
  scp_dir = 'data/kaldi-scp'
  out_dir = 'data/seg_data'
  SS = ScpSegment(scp_dir, out_dir)
  SS.seg_data(40)


