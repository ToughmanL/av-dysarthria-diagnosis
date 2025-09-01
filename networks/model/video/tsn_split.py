# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

from networks.model.ops.basic_ops import ConsensusModule
from networks.model.ops.transforms import *
from torch.nn.init import normal_, constant_
import random
from networks.model.ops.temporal_prep import get_rgbdiff, get_flow, get_paflow

random.seed(1)

# random select index
def select_index(data, num_segments, rgb_len=1):
  rgb_data_list, temporal_data_list = [], []
  for i in range(num_segments):
    if isinstance(data, list):
      seg_data = data[i]
    elif isinstance(data, torch.Tensor):
      seg_data = data[:, i, ...]
    start, end = int(seg_data.size(1) * 0.1), int(seg_data.size(1) * 0.9)

    # random_indices = random.sample(range(start, end + 1), rgb_len)
    # frames = [seg_data[:, idx] for idx in random_indices]
    # select_data = torch.stack(frames, dim=1) # 随机选
    
    select_data = seg_data[:, int((start+end)/2)] # 选中间

    rgb_data_list.append(select_data)
    temporal_data_list.append(seg_data)
  rgb_data = torch.stack(rgb_data_list, dim=1)
  temporal_data = torch.cat(temporal_data_list, dim=1)
  return (rgb_data, temporal_data)


class TSNSPLIT(nn.Module):
  def __init__(self, out_dim, num_segments, modality,
         base_model='resnet50', new_length=61,
         consensus_type='avg', before_softmax=True,
         dropout=0.4, img_feature_dim=256,
         crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
         is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
         temporal_pool=False, non_local=False, window_size=1):
    super(TSNSPLIT, self).__init__()
    self.modality = modality
    if num_segments < 0:
      self.ait_segment = True
    else:
      self.ait_segment = False
    self.num_segments = abs(num_segments)
    self.reshape = True
    self.before_softmax = before_softmax
    self.dropout = dropout
    self.crop_num = crop_num
    self.consensus_type = consensus_type
    self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
    self.pretrain = pretrain
    self.window_size = window_size

    self.is_shift = is_shift
    self.shift_div = shift_div
    self.shift_place = shift_place
    self.base_model_name = base_model
    self.fc_lr5 = fc_lr5
    self.temporal_pool = temporal_pool
    self.non_local = non_local

    if not before_softmax and consensus_type != 'avg':
      raise ValueError("Only avg consensus can be used after Softmax")

    self.new_length = new_length
    if print_spec:
      print(("""
            Initializing TSNSPLIT with base model: {}.
            TSNSPLIT Configurations:
              input_modality:   {}
              num_segments:     {}
              new_length:     {}
              consensus_module:   {}
              dropout_ratio:    {}
              img_feature_dim:  {}
      """.format(base_model, modality, self.num_segments, self.new_length,  consensus_type, self.dropout, self.img_feature_dim)))

    self._prepare_base_model(base_model)

    feature_dim = self._prepare_tsn(out_dim)
    if self.window_size > 1:
      print("Converting the ImageNet model to Split init model")
      self.base_model = self._construct_split_model(self.base_model, self.window_size)
      print("Done. Split model ready.")

    self.consensus = ConsensusModule(consensus_type)

    if not self.before_softmax:
      self.softmax = nn.Softmax()

    self._enable_pbn = partial_bn
    if partial_bn:
      self.partialBN(True)
    
  def _prepare_tsn(self, out_dim):
    feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
    if self.dropout == 0:
      setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, out_dim))
      self.new_fc = None
    else:
      setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
      self.new_fc = nn.Linear(feature_dim, out_dim)

    std = 0.001
    if self.new_fc is None:
      normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
      constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
    else:
      if hasattr(self.new_fc, 'weight'):
        normal_(self.new_fc.weight, 0, std)
        constant_(self.new_fc.bias, 0)
    return feature_dim

  def _prepare_base_model(self, base_model):
    print('=> base model: {}'.format(base_model))

    if 'resnet' in base_model:
      self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
      if self.is_shift:
        print('Adding temporal shift...')
        from networks.model.ops.temporal_shift import make_temporal_shift
        make_temporal_shift(self.base_model, self.num_segments,
                  n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

      if self.non_local:
        print('Adding non-local module...')
        from networks.model.opts.non_local import make_non_local
        make_non_local(self.base_model, self.num_segments)

      self.base_model.last_layer_name = 'fc'
      self.input_size = 96
      self.input_mean = [0.485, 0.456, 0.406]
      self.input_std = [0.229, 0.224, 0.225]

      self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

      if 'Flow' == self.modality:
        self.input_mean = [0.5]
        self.input_std = [np.mean(self.input_std)]
      elif 'RGBDiff' == self.modality:
        self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
        self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

    elif base_model == 'mobilenetv2':
      from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
      self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)

      self.base_model.last_layer_name = 'classifier'
      self.input_size = 96
      self.input_mean = [0.485, 0.456, 0.406]
      self.input_std = [0.229, 0.224, 0.225]

      self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
      if self.is_shift:
        from networks.model.ops.temporal_shift import TemporalShift
        for m in self.base_model.modules():
          if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
            if self.print_spec:
              print('Adding temporal shift... {}'.format(m.use_res_connect))
            m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
      if 'Flow' == self.modality:
        self.input_mean = [0.5]
        self.input_std = [np.mean(self.input_std)]
      elif 'RGBDiff' == self.modality:
        self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
        self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

    elif base_model == 'BNInception':
      from archs.bn_inception import bninception
      self.base_model = bninception(pretrained=self.pretrain)
      self.input_size = self.base_model.input_size
      self.input_mean = self.base_model.mean
      self.input_std = self.base_model.std
      self.base_model.last_layer_name = 'fc'
      if 'Flow' == self.modality:
        self.input_mean = [128]
      elif 'RGBDiff' == self.modality:
        self.input_mean = self.input_mean * (1 + self.new_length)
      if self.is_shift:
        print('Adding temporal shift...')
        self.base_model.build_temporal_ops(
          self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
    else:
      raise ValueError('Unknown base model: {}'.format(base_model))

  def train(self, mode=True):
    """
    Override the default train() to freeze the BN parameters
    :return:
    """
    super(TSNSPLIT, self).train(mode)
    count = 0
    if self._enable_pbn and mode:
      # print("Freezing BatchNorm2D except the first one.")
      for m in self.base_model.modules():
        if isinstance(m, nn.BatchNorm2d):
          count += 1
          if count >= (2 if self._enable_pbn else 1):
            m.eval()
            # shutdown update in frozen mode
            m.weight.requires_grad = False
            m.bias.requires_grad = False

  def partialBN(self, enable):
    self._enable_pbn = enable

  def get_optim_policies(self):
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    lr5_weight = []
    lr10_bias = []
    bn = []
    custom_ops = []

    conv_cnt = 0
    bn_cnt = 0
    for m in self.modules():
      if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
        ps = list(m.parameters())
        conv_cnt += 1
        if conv_cnt == 1:
          first_conv_weight.append(ps[0])
          if len(ps) == 2:
            first_conv_bias.append(ps[1])
        else:
          normal_weight.append(ps[0])
          if len(ps) == 2:
            normal_bias.append(ps[1])
      elif isinstance(m, torch.nn.Linear):
        ps = list(m.parameters())
        if self.fc_lr5:
          lr5_weight.append(ps[0])
        else:
          normal_weight.append(ps[0])
        if len(ps) == 2:
          if self.fc_lr5:
            lr10_bias.append(ps[1])
          else:
            normal_bias.append(ps[1])

      elif isinstance(m, torch.nn.BatchNorm2d):
        bn_cnt += 1
        # later BN's are frozen
        if not self._enable_pbn or bn_cnt == 1:
          bn.extend(list(m.parameters()))
      elif isinstance(m, torch.nn.BatchNorm3d):
        bn_cnt += 1
        # later BN's are frozen
        if not self._enable_pbn or bn_cnt == 1:
          bn.extend(list(m.parameters()))
      elif len(m._modules) == 0:
        if len(list(m.parameters())) > 0:
          raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    return [
      {'params': first_conv_weight, 'lr_mult': 5 if  'Flow' == self.modality else 1, 'decay_mult': 1,
       'name': "first_conv_weight"},
      {'params': first_conv_bias, 'lr_mult': 10 if  'Flow' == self.modality else 2, 'decay_mult': 0,
       'name': "first_conv_bias"},
      {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
       'name': "normal_weight"},
      {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
       'name': "normal_bias"},
      {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
       'name': "BN scale/shift"},
      {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
       'name': "custom_ops"},
      # for fc
      {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
       'name': "lr5_weight"},
      {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
       'name': "lr10_bias"},
    ]

  # segment data and two stream
  def segment_data(self, input):
    if self.num_segments == 1:
      # 在第二维增加一个维度
      input = input.unsqueeze(1)
      return select_index(input, 1)
    elif self.ait_segment:
      # 将数据分为四段，第一段为静音段10帧，第二段为AIT段15帧，第四段为后段7帧，中间段为剩余帧
      front_len, ait_len, back_len = 10, 15, 7
      data_len = input.size(1)
      if data_len < front_len + ait_len + back_len:
        raise ValueError("Data length is not enough")
      front_data = input[:, :front_len]
      ait_data = input[:, front_len:front_len + ait_len]
      middle_data = input[:, front_len + ait_len:-back_len]
      back_data = input[:, -back_len:]
      input_data = [front_data, ait_data, middle_data, back_data]
      return select_index(input_data, 4)
    else:
      chunk_data = input.chunk(self.num_segments, 1)
      input_data = torch.stack(chunk_data, dim=1)
      return select_index((input_data), self.num_segments)
  
  def window_partition(self, x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3).contiguous()
    windows = windows.reshape(B, C*window_size, H, W // window_size)
    return windows

  def extract_feats(self, input):
    batch_size, length, channel, height, width = input.size()
    rgb_data, flow_data = self.segment_data(input)
    temp_data, spatial_data = None, None
    temp_out, spatial_out = None, None
    
    # rgb
    spatial_data = rgb_data.view(batch_size, channel*self.num_segments, height, width)
    spatial_data = self.window_partition(spatial_data, self.window_size)

    if temp_data is not None:
      temp_out = self.base_model(temp_data)
      temp_out = temp_out.view(batch_size, self.num_segments, -1)
      temp_out = self.consensus(temp_out)
      temp_out = torch.squeeze(temp_out, 1)

    if spatial_data is not None:
      spatial_out = self.base_model(spatial_data)
      spatial_out = spatial_out.view(batch_size, self.num_segments, -1)
      spatial_out = self.consensus(spatial_out)
      spatial_out = torch.squeeze(spatial_out, 1)

    temp_spatial_outs = [outdata for outdata in [temp_out, spatial_out] if outdata is not None]
    return torch.cat(temp_spatial_outs, 1)

  def forward(
        self,
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lengths: torch.Tensor
    ):
  # def forward(
  #       self,
  #       video: torch.Tensor,
  #   ):
    input = video
    base_out = self.extract_feats(input)

    if self.dropout > 0:
      base_out = self.new_fc(base_out)
    if not self.before_softmax:
      base_out = self.softmax(base_out)

    return base_out.squeeze(1)

  def _construct_split_model(self, base_model, window_size):
    # modify the convolution layers
    # Torch models are usually defined in a hierarchical way.
    # nn.modules.children() return all sub modules in a DFS manner
    modules = list(self.base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (window_size*3, ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv2d(window_size*3, conv_layer.out_channels,
                          conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                          bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)

    return base_model

# test tsn
if __name__ == "__main__":
  import torch
  import torchvision

  base_model = 'resnet50'
  out_dim = 64
  num_segments = 1
  modality = 'RGB'
  consensus_type = 'avg'
  dropout = 0.6
  img_feature_dim = 256
  crop_num = 1
  partial_bn = True
  pretrain = 'imagenet'
  is_shift = False
  shift_div = 8
  shift_place = 'blockres'
  fc_lr5 = False
  temporal_pool = False
  non_local = False

  b, t, c, h, w =64, 61, 3, 96, 96
  input = torch.randn(b, t, c, h, w) # btchw

  tsn = TSNSPLIT(out_dim, num_segments, modality, base_model, new_length=t,consensus_type=consensus_type, dropout=dropout,
       img_feature_dim=img_feature_dim, crop_num=crop_num, partial_bn=partial_bn, pretrain=pretrain,
       is_shift=is_shift, shift_div=shift_div, shift_place=shift_place, fc_lr5=fc_lr5, temporal_pool=temporal_pool,
       non_local=non_local)

  output = tsn(input)
  print(output.size())
  print(output)
  print('TSNSPLIT test passed.')
