#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 feat_select.py
* @Time 	:	 2023/08/15
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025'','' lxk&AISML
* @Desc   	:	 None
'''
import os
import pandas as pd

common_feats = ['Jitter','Shimmer','HNR','gne','vfer','F1_sd','F2_sd','F3_sd','Intensity_mean','Intensity_sd','Vowel_dur','Syllable_dur','gop_con','gop_vow']
arti_feats = ['tougue_dist','jaw_dist','move_degree','VSA','VAI','FCR']

gop_feats = ['gop_con','gop_vow']
LiuYY_arti_feats = ['move_degree','VSA','VAI','FCR']

l_fets = ['alpha_stability','beta_stability','dist_stability','alpha_speed','beta_speed','inner_speed','inner_dist_min','inner_dist_max','w_min','w_max']

labels = ['Frenchay','fanshe','huxi','chun','he','ruane','hou','she','yanyu','ljri', 'rlvt']

mfcc_feats = ['Person','u_mfcc','i_mfcc','a_mfcc','e_mfcc','o_mfcc','v_mfcc','Frenchay','fanshe','huxi','chun','he','ruane','hou','she','yanyu']

egemaps_feats = ['Person','u_egemaps','i_egemaps','a_egemaps','e_egemaps','o_egemaps','v_egemaps','Frenchay','fanshe','huxi','chun','he','ruane','hou','she','yanyu']

wavsegment_feats = ['Path','Start','End']
wav2vecsegment_feats = ['Segname']
hubertsegment_feats = ['Segname']
srft_feats = ['Path','Start','End']


def get_feat_name(feat_type, vowels):
  sel_feats_name = ['Person']
  if feat_type == 'loop_featnorm' or feat_type == 'gop_loop_featnorm':
    vowel_common = []
    for vowel in vowels:
      for feat in common_feats:
        vowel_common.append(vowel + '-' + feat)
    sel_feats_name = sel_feats_name + arti_feats + vowel_common + labels
  elif feat_type == 'loop_v_featnorm':
    vowel_lip = []
    for vowel in vowels:
      for feat in l_fets:
        vowel_lip.append(vowel + '-' + feat)
    sel_feats_name = sel_feats_name + vowel_lip + labels
  elif feat_type == 'loop_av_featnorm':
    vowel_papil = []
    for vowel in vowels:
      for feat in common_feats:
        vowel_papil.append(vowel + '-' + feat)
      for feat in l_fets:
        vowel_papil.append(vowel + '-' + feat)
    sel_feats_name = sel_feats_name + arti_feats + vowel_papil + labels
  elif feat_type == 'mfcc':
    vowel_mfcc = []
    for vowel in vowels:
      vowel_mfcc.append(vowel + '_' + 'mfcc')
    sel_feats_name = sel_feats_name + vowel_mfcc + labels
  elif feat_type == 'egemaps':
    vowel_egemaps = []
    for vowel in vowels:
      vowel_egemaps.append(vowel + '_' + 'egemaps')
    sel_feats_name = sel_feats_name + vowel_egemaps + labels
  elif feat_type == 'gop':
    vowel_gop = []
    for vowel in vowels:
      for feat in gop_feats:
        vowel_gop.append(vowel + '-' + feat)
    sel_feats_name = sel_feats_name + vowel_gop + labels
  elif feat_type == 'Liuartifeat':
    sel_feats_name = sel_feats_name + LiuYY_arti_feats + labels
  elif feat_type == 'wavsegment':
    sel_feats_name = sel_feats_name + wavsegment_feats + labels
  elif feat_type == 'stft':
    sel_feats_name = sel_feats_name + srft_feats + labels
  elif feat_type == 'papi_cmlrv':
    vowel_papil = []
    vowel_mfcc = []
    for vowel in vowels:
      for feat in common_feats:
        vowel_papil.append(vowel + '-' + feat)
    for vowel in vowels:
      vowel_mfcc.append(vowel + '_' + 'mfcc')
    sel_feats_name = sel_feats_name + arti_feats + vowel_papil + vowel_mfcc + labels
  elif feat_type == 'wav2vecsegment':
    sel_feats_name = sel_feats_name + wav2vecsegment_feats + labels
  elif feat_type == 'hubertsegment':
    sel_feats_name = sel_feats_name + hubertsegment_feats + labels
  return sel_feats_name

def feat_sel(feat_type, vowels, featsdf):
  feats_name = get_feat_name(feat_type, vowels)
  sel_feat = featsdf[feats_name]
  return sel_feat


def get_file_path(setment_dir, file_name, suffix):
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
  file_path = os.path.join(setment_dir, N_S_name, person_name, file_name + '.' + suffix)
  return file_path


