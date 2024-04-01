#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 init_model.py
* @Time 	:	 2024/03/19
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

from networks.model.resnet import ResNetStft 
from networks.model.lstm import LSTMNet 
# from networks.utils.cmvn import load_cmvn
from networks.utils.checkpoint import load_checkpoint, load_trained_modules

MODEL_CLASSES = {
    "resnet": ResNetStft,
    "lstm": LSTMNet
}

# init model
def init_model(configs):
    model_type = configs.get('model_name')
    input_dim = configs['input_dim']
    hidden_dim = configs['hidden_dim']
    output_dim = configs['output_dim']
    
    model = MODEL_CLASSES[model_type](
        input_dim,
        hidden_dim,
        output_dim)

    # If specify checkpoint, load some info from checkpoint
    if 'checkpoint' in configs and configs['checkpoint'] is not None:
        infos = load_checkpoint(model, configs.checkpoint)
    elif 'enc_init' in configs and configs['enc_init'] is not None:
        infos = load_trained_modules(model, configs)
    else:
        infos = {}
    configs["init_infos"] = infos
    print(configs)
    return model, configs

