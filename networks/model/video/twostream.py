# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
from torch import nn
from networks.model.video.tsn import TSN


class TWOSTREAM(nn.Module):
  def __init__(self, num_class, out_dim, num_segments, modality,
         time_base_model='resnet50', space_base_model='resnet50',
         new_length=61, dropout=0.65):
    super(TWOSTREAM, self).__init__()
    self.modality = modality.split("+")
    self.space_branch = TSN(out_dim, num_segments, self.modality[0], space_base_model, 1, dropout=dropout)
    self.time_branch = TSN(out_dim, num_segments, self.modality[1], time_base_model, new_length, dropout=dropout)
    self.multi_attn = nn.MultiheadAttention(1, 1, dropout=dropout)
    
    self.relu = nn.ReLU(inplace=False)
    self.fc = nn.Linear(out_dim * 2, num_class)
    self.dropout = nn.Dropout(dropout)
  
  def forward(
        self,
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lengths: torch.Tensor
    ):
    speech = torch.empty(0)
    speech_lengths = torch.empty(0)
    input= video
    rgb_out = self.space_branch(speech, input, speech_lengths)
    flow_out = self.time_branch(speech, input, speech_lengths)
    output = torch.cat([rgb_out, flow_out], 1)
    output_emb = output.view(output.size(1), output.size(0), -1)
    output, weights = self.multi_attn(output_emb, output_emb, output_emb)
    output = output.view(output.size(1), output.size(0))
    output = self.fc(self.dropout(output))
    return output.squeeze(1)

# test twostream
if __name__ == "__main__":
  
  model = TWOSTREAM(2, 64, 1, 'RGB+RGBDiff', 'resnet50', 'resnet50', 61, 0.5)
  input = torch.randn(64, 61, 3, 96, 96)
  output = model(input)
  print(output.shape)
  print(output)
  print("test twostream pass")