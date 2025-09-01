import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.model.ops.basic_ops import ModelFusion

from collections import OrderedDict
import os


def pretrain_model(model: torch.nn.Module, path: str):
  # Load encoder modules with pre-trained model(s).
  main_state_dict = model.state_dict()
  partial_state_dict = OrderedDict()
  key_value_match = {'ShapMatch':[], 'ShapeMismatch':[], 'KeyNotFound':[]}
  print("model(s) found for pre-initialization")
  if os.path.isfile(path):
      print('Checkpoint:  %s ' % path)
      model_state_dict = torch.load(path, map_location='cpu')
      for key, value in model_state_dict.items():
          if key in main_state_dict:
              if value.shape == main_state_dict[key].shape:
                  key_value_match['ShapMatch'] += [key]
                  partial_state_dict[key] = value
              else:
                  key_value_match['ShapeMismatch'] += [key]
                  partial_state_dict[key] = main_state_dict[key]
          else:
              key_value_match['KeyNotFound'] += [key]
  else:
      print("model was not found : %s", path)

  print("%d Key(s) not found in model" % len(key_value_match['KeyNotFound']))
  print("%d Key(s) with mismatched shape" % len(key_value_match['ShapeMismatch']))
  print("%d Key(s) with matched shape" % len(key_value_match['ShapMatch']))

  model.load_state_dict(partial_state_dict, strict=False)
  return model


class MultiLoss(nn.Module):
  def __init__(self, 
               fold: int,
               audio_net: torch.nn.Module, 
               video_net: torch.nn.Module,
               audio_dim: int,
               video_dim: int,
               fusion_type: str,
               output_size: int):
    super().__init__()
    self.audio_net = audio_net
    self.visual_net = video_net
    self.audio_net = pretrain_model(self.audio_net, 'exp/resnetseq10_300f_linear/models/classification_fold_' + str(fold) + '_best.pt')
    self.visual_net = pretrain_model(self.visual_net, 'exp/ts_rgbpaflow_TSN_seg-4split4/models/classification_fold_' + str(fold) + '_best.pt')
    self.fusion_net = ModelFusion(fusion_type, audio_dim, dim=1)
    if fusion_type == 'concat':
      av_dim = audio_dim + video_dim
    else:
      av_dim = audio_dim
    
    self.relu = nn.ReLU()
    self.a_bn_1d = nn.BatchNorm1d(audio_dim)
    self.v_bn_1d = nn.BatchNorm1d(video_dim)
    self.head_audio = nn.Linear(audio_dim, output_size)
    self.head_video = nn.Linear(video_dim, output_size)
    self.head = nn.Linear(av_dim, output_size)
  
  def forward(self, 
      speech: torch.Tensor,
      video: torch.Tensor,
      text: torch.Tensor,
      speech_lens: torch.Tensor,
      text_lens: torch.Tensor):
    a = self.audio_net(speech, speech_lens)
    v = self.visual_net(speech, video, speech_lens, text)

    out_audio = self.head_audio(a)
    out_video = self.head_video(v)

    a = self.a_bn_1d(self.relu(a))
    v = self.v_bn_1d(self.relu(v))

    # a = self.relu(a)
    # v = self.relu(v)
    
    out = self.fusion_net(a, v)
    out = self.head(out)
    
    return out, out_audio, out_video




if __name__ == '__main__':
  import yaml
  from networks.model.audio_resnet.resnetsqu import get_resnetsqu
  from networks.model.video.tsn_seg_split import TSNSEGSPLIT

  fold = 0
  config_file = 'conf/av_resnetseq10_tsnsegsplit_mtlcroatt.yaml'
  with open(config_file, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
  audio_conf = configs['audio_conf']
  visual_conf = configs['visual_conf']
  audio_model = get_resnetsqu(resnum=audio_conf['resnum'], feat_dim=audio_conf['input_dim'], embed_dim=audio_conf['embed_dim'])
  visual_model = TSNSEGSPLIT(
      num_class=configs['task_info']['score'],
      out_dim=visual_conf['out_dim'],
      num_segments=visual_conf['num_segments'],
      modality=visual_conf['modality'],
      new_length=visual_conf['new_length'],
      consensus_type=visual_conf['consensus_type'],
      window_size=visual_conf['window_size'],
      dropout=visual_conf['dropout'])
  model = AVClassifier(fold, audio_model, visual_model, audio_conf['embed_dim'], visual_conf['out_dim'], configs['av_fusion_type'], configs['output_dim'])

  speech = torch.zeros(64, 200, 80)
  speech_len = torch.tensor([64, 200])
  video_data = torch.zeros([64, 61, 3, 96, 96])
  text_data = torch.zeros([64, 114])
  text_len = torch.tensor([64, 2])
  label = torch.tensor([64])

  out, out_audio, out_video = model(speech, video_data, text_data, speech_len, text_len, label)
  print(out.shape, out_audio.shape, out_video.shape)