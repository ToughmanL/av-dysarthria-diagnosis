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

class LSTMNet(nn.Module):
  def __init__(self,input_dim,hidden_dim,output_dim,n_layers=2,bidirectional=True,dropout=0.2):
    super(LSTMNet,self).__init__()
    # LSTM layer process the vector sequences 
    self.lstm = nn.LSTM(input_dim,
                        hidden_dim,
                        num_layers = n_layers,
                        bidirectional = bidirectional,
                        dropout = dropout,
                        batch_first = True
                        )
    # Dense layer to predict 
    self.fc = nn.Linear(hidden_dim * 2,output_dim)
    # Prediction activation function
    self.sigmoid = nn.Sigmoid()
        
    
  def forward(self,feat):
    # Packing the padded sequence
    # packed_feat = nn.utils.rnn.pack_padded_sequence(feat, feat_lengths.cpu(),batch_first=True)
    packed_output,(hidden_state,cell_state) = self.lstm(feat)
    # Concatenating the final forward and backward hidden states
    hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
    dense_outputs=self.fc(hidden)
    #Final activation function
    outputs=self.sigmoid(dense_outputs)
    return outputs

# test
if __name__ == '__main__':
  input_dim, hidden_dim, output_dim = 768, 64, 4
  model = LSTMNet(input_dim,hidden_dim,output_dim)
  print(model)
  feat_data = torch.randn(2, 51, 768)
  output = model.forward(feat_data, torch.tensor([51, 51]))
  print(output.size())