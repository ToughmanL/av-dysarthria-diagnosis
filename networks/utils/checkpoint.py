#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	checkpoint.py
 *@Time	: 2023-12-25 23:59:47
 *@Author	:	lxk
 *@Version	:	1.2
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''


import logging
import os
import re
import yaml
import torch
from collections import OrderedDict


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
  checkpoint = torch.load(path, map_location='cpu')
  model.load_state_dict(checkpoint, strict=False)
  info_path = re.sub('.pt$', '.yaml', path)
  configs = {}
  if os.path.exists(info_path):
    with open(info_path, 'r') as fin:
      configs = yaml.load(fin, Loader=yaml.FullLoader)
  return configs

def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
  '''
  Args:
    infos (dict or None): any info you want to save.
  '''
  logging.info('Checkpoint: save to checkpoint %s' % path)
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save(state_dict, path)
  # info_path = re.sub('.pt$', '.yaml', path)
  # if infos is None:
  #   infos = {}
  # with open(info_path, 'w') as fout:
  #   data = yaml.dump(infos)
  #   fout.write(data)

def migration(model: torch.nn.Module, path: str):
  source_dict = model.state_dict()
  target_dict = torch.load(path)
  migrat_dict = {}
  for k, v in source_dict.items():
    if k in target_dict.keys():
      if source_dict[k].shape == target_dict[k].shape:
        migrat_dict[k] = v
      else:
        print("Inconsistent parameters {}".format(k))
  model.load_state_dict(migrat_dict, strict=False)
  return model

def filter_modules(model_state_dict, modules):
  new_mods = []
  incorrect_mods = []
  mods_model = model_state_dict.keys()
  for mod in modules:
    if any(key.startswith(mod) for key in mods_model):
      new_mods += [mod]
    else:
      incorrect_mods += [mod]
  if incorrect_mods:
    logging.warning(
      "module(s) %s don't match or (partially match) "
      "available modules in model.",
      incorrect_mods,
    )
    logging.warning("for information, the existing modules in model are:")
    logging.warning("%s", mods_model)

  return new_mods

def load_trained_modules(model: torch.nn.Module, args: None):
  # Load encoder modules with pre-trained model(s).
  enc_model_path = args.enc_init
  enc_modules = args.enc_init_mods
  main_state_dict = model.state_dict()
  logging.warning("model(s) found for pre-initialization")
  if os.path.isfile(enc_model_path):
    logging.info('Checkpoint: loading from checkpoint %s for CPU' %
                  enc_model_path)
    model_state_dict = torch.load(enc_model_path, map_location='cpu')
    modules = filter_modules(model_state_dict, enc_modules)
    partial_state_dict = OrderedDict()
    for key, value in model_state_dict.items():
      if any(key.startswith(m) for m in modules):
        partial_state_dict[key] = value
    main_state_dict.update(partial_state_dict)
  else:
    logging.warning("model was not found : %s", enc_model_path)

  model.load_state_dict(main_state_dict)
  configs = {}
  return configs