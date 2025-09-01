#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 lstm.py
* @Time 	:	 2024/03/21
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMNet(nn.Module):
  def __init__(self, 
      input_dim,
      hidden_dim,
      output_dim,
      n_layers=2,
      bidirectional=True,
      dropout=0.3):
    super(LSTMNet,self).__init__()
    self.out_dim = output_dim
    # LSTM layer process the vector sequences 
    self.lstm = nn.LSTM(input_dim,
                        hidden_dim,
                        num_layers = n_layers,
                        bidirectional = bidirectional,
                        dropout = dropout,
                        batch_first = True
                        )
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.dropout = nn.Dropout(dropout)


  def forward(self, 
        speech: torch.Tensor,
        video: torch.Tensor,
        speech_lens: torch.Tensor
    ):
    speech_lens = speech_lens.cpu().numpy()
    packed_input = pack_padded_sequence(speech, speech_lens, batch_first=True, enforce_sorted=False)
    packed_output, (hidden_state, cell_state) = self.lstm(packed_input)
    # output, _ = pad_packed_sequence(packed_output, batch_first=True)
    # hidden = torch.cat((hidden_state[-2,:,:],hidden_state[-1,:,:]),dim=1)
    outputs = self.fc(hidden_state[-1,:,:])
    outputs = self.dropout(outputs)
    return outputs

  # def forward(self, feat):
  #   packed_output,(hidden_state,cell_state) = self.lstm(feat)
  #   hidden = torch.cat((hidden_state[-2,:,:],hidden_state[-1,:,:]),dim=1)
  #   outputs=self.fc(hidden)
  #   return outputs
  
  def output_size(self):
    return self.out_dim

# test
if __name__ == '__main__':
  input_dim, hidden_dim, output_dim = 768, 64, 4
  model = LSTMNet(input_dim,hidden_dim,output_dim)
  print(model)
  feat_data = torch.randn(2, 51, 768)
  output = model.forward(feat_data, torch.tensor([51, 51]))
  print(output.size())