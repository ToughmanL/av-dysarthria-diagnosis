#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 trainer.py
* @Time 	:	 2023/02/26
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from networks.utils.basetrainer import BaseTrainer
from networks.utils.checkpoint import save_checkpoint
from networks.utils.early_stop import EarlyStopping
from networks.utils.min_norm_solvers import MinNormSolver
from surgeon_pytorch import Inspect, get_layers
from tools.evaluation import result_process, classification_eval

from networks.utils.loss_fun import MultiTaskLoss


class NNTrainer(BaseTrainer):
  def __init__(self, configs, label, model, loss_fn, optimizer, scheduler):
    super().__init__(configs, label, model, loss_fn, optimizer, scheduler)
    self.accum_grad = 1 if 'accum_grad' not in self.configs else self.configs['accum_grad']
    # 分类还是回归
    self.task_type = self.configs['task_info']['type']
    self.earlystop = self.configs['earlystop']
    self.n_epochs = self.configs['EPOCHS']

  def train_step(self, x, y):
    self.model.train()
    yhat = self.model(x)
    if self.task_type == 'classification':
      loss = self.loss_fn(yhat, y.to(torch.int64))
    elif self.task_type == 'regression':
      loss = self.loss_fn(yhat, y.unsqueeze(1))
    loss = loss/self.accum_grad
    loss.backward()
    return loss.item()

  def train(self, train_loader, val_loader, device, fold_i):
    if self.earlystop > 0:
      early_stopping = EarlyStopping(patience=self.earlystop, verbose=True)
    train_info_list = []
    best_valid_loss = 10000

    for epoch in range(1, self.n_epochs + 1):
      batch_losses = []
      # 训练
      for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        loss = self.train_step(x_batch, y_batch)
        if ((epoch) % self.accum_grad == 0) or (epoch == self.n_epochs):
          self.optimizer.step()
          self.optimizer.zero_grad()
        if np.isnan(loss):
          print('nan loss')
        else:
          batch_losses.append(loss)
      self.scheduler.step()
      training_loss = np.mean(batch_losses)
      self.train_losses.append(training_loss)

      # 验证
      with torch.no_grad():
        batch_val_losses = []
        for x_val, y_val in val_loader:
          x_val, y_val = x_val.to(device), y_val.to(device)
          self.model.eval()
          yhat = self.model(x_val)
          if self.task_type == 'classification':
            val_loss = self.loss_fn(yhat, y_val.to(torch.int64)).item()
          elif self.task_type == 'regression':
            val_loss = self.loss_fn(yhat, y_val.unsqueeze(1)).item()
          if np.isnan(val_loss):
            print('nan loss')
          else:
            batch_val_losses.append(val_loss)
        validation_loss = np.mean(batch_val_losses)
        self.val_losses.append(validation_loss)

      # 保存每个epoch模型
      if ((epoch) % self.accum_grad == 0) or (epoch == self.n_epochs):
        model_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_{str(epoch)}.pt"
        dt_now = datetime.datetime.now()
        h_m_s = "{}:{}:{}".format(dt_now.strftime('%H'), dt_now.strftime('%M'),dt_now.strftime('%S'))
        train_info = f"| {h_m_s} | Fold: {fold_i} | Epoch: {'%03d'%epoch} | TraLoss: {'%.2f'%training_loss} | ValLoss |: {'%.2f'%validation_loss} | LR: {'%.6f'%self.scheduler._last_lr[0]}"
        print(train_info)
        train_info_list.append({'Fold':fold_i, 'Epoch':epoch, 'TrainLoss':training_loss, 'ValLoss':validation_loss})

        # 保存最好的模型
        if validation_loss < best_valid_loss:
          best_valid_loss = validation_loss
          best_model_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_best.pt"
          save_checkpoint(self.model, model_path)
          save_checkpoint(self.model, best_model_path)

        # 早停机制，为避免过早停，在10轮之后计数
        if self.earlystop and epoch > 10*self.accum_grad:
          early_stopping(validation_loss)
          if early_stopping.early_stop:
            print("Early stopping")
            break

    # 训练过程可视化
    train_val_loss_df = pd.DataFrame.from_dict({'train':self.train_losses, 'val':self.val_losses}).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    # plt.figure(figsize=(15,8))
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.savefig(f"{self.configs['model_dir']}/pics/{self.configs['model_name']}_{self.label}_fold{str(fold_i)}.png")
    plt.clf()

    # 保存训练数据
    info_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_loss.csv"
    df = pd.DataFrame(train_info_list)
    df.to_csv(info_path, index=False)

  def evaluate(self, test_loader, device):
    with torch.no_grad():
      predictions = []
      values = []
      for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        self.model.eval()
        yhat = self.model(x_test)
        if self.task_type == 'classification':
          yhat = torch.argmax(yhat, dim=1)
        predictions.append(yhat.cpu().detach().numpy())
        values.append(y_test.unsqueeze(1).cpu().detach().numpy())
    return predictions, values

  def get_middle_layer(self, test_loader, layername, device):
    with torch.no_grad():
      middle_list = []
      values = []
      relu_out = None
      for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        self.model.eval()
        model_wrapped = Inspect(self.model, layer=layername)
        y, x2 = model_wrapped(x_test)
        feat_label = torch.cat((x2.squeeze(), y.squeeze(0), y_test.unsqueeze(1)), dim=1) # 最后两维分别是预测值和目标值
        middle_list.append(feat_label.cpu().detach().numpy())
    middle_data = np.concatenate(middle_list, axis=0)
    column_names = ['feat_{}'.format(str(i)) for i in range(middle_data.shape[1]-2)]
    column_names.append('Predict')
    column_names.append('Target')
    df_middle_data = pd.DataFrame(middle_data, columns=column_names)
    return df_middle_data


class SpeechTrainer(BaseTrainer):
  def __init__(self, configs, label, model, loss_fn, optimizer, scheduler):
    super().__init__(configs, label, model, loss_fn, optimizer, scheduler)
    self.accum_grad = 1 if 'accum_grad' not in self.configs else self.configs['accum_grad']
    # 分类还是回归
    self.task_type = self.configs['task_info']['type']
    self.earlystop = self.configs['earlystop']
    self.n_epochs = self.configs['EPOCHS']
    self.batch_size = self.configs['batch_size']

  def train_step(self, audio_feats, video_feats, feats_lens, video_len, target):
    self.model.train()
    yhat = self.model(audio_feats, video_feats, feats_lens)
    if self.task_type == 'classification':
      loss = self.loss_fn(yhat, target.squeeze(1).to(torch.int64))
    elif self.task_type == 'regression':
      loss = self.loss_fn(yhat, target.unsqueeze(1))
    loss = loss/self.accum_grad
    loss.backward()
    return loss.item(), yhat

  def train(self, train_loader, val_loader, device, fold_i):
    if self.earlystop > 0:
      early_stopping = EarlyStopping(patience=self.earlystop, verbose=True)
    train_info_list = []
    best_valid_loss = 10000

    for epoch in range(1, self.n_epochs + 1):
      batch_losses = []
      keys, predictions, values = [], [], []
      # 训练
      for batch_idx, batch in enumerate(train_loader):
        key, audio_feats, video_feats, target, feats_lengths, video_len = batch
        audio_feats = audio_feats.to(device)
        video_feats = video_feats.to(device)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        video_len = video_len.to(device)
        loss, yhat = self.train_step(audio_feats, video_feats, feats_lengths, video_len, target)
        predictions.extend(torch.argmax(yhat, dim=1).cpu().detach().tolist())
        if ((batch_idx) % self.accum_grad == 0):
          self.optimizer.step()
          self.optimizer.zero_grad()
        if np.isnan(loss):
          print('nan loss')
        else:
          batch_losses.append(loss)
        keys.extend(key)
        values.extend(target.cpu().detach().tolist())
      train_results = (keys, predictions, values)
      self.scheduler.step()
      training_loss = np.mean(batch_losses)
      self.train_losses.append(training_loss*self.accum_grad)

      # 验证
      with torch.no_grad():
        batch_val_losses = []
        keys, predictions, values = [], [], []
        for batch_idx, batch in enumerate(val_loader):
          key, audio_feats, video_feats, target, feats_lengths, video_len = batch
          audio_feats = audio_feats.to(device)
          video_feats = video_feats.to(device)
          target = target.to(device)
          feats_lengths = feats_lengths.to(device)
          video_len = video_len.to(device)
          self.model.eval()
          yhat = self.model(audio_feats, video_feats, feats_lengths)
          if self.task_type == 'classification':
            val_loss = self.loss_fn(yhat, target.squeeze(1).to(torch.int64))
            predictions.extend(torch.argmax(yhat, dim=1).cpu().detach().tolist())
          elif self.task_type == 'regression':
            val_loss = self.loss_fn(yhat, target.unsqueeze(1)).item()
          val_loss = val_loss.cpu().numpy()
          values.extend(target.cpu().detach().tolist())
          keys.extend(key)
          if np.isnan(val_loss):
            print('nan loss')
          else:
            batch_val_losses.append(val_loss)
        test_results = (keys, predictions, values)
        validation_loss = np.mean(batch_val_losses)
        self.val_losses.append(validation_loss)

      # 保存每个epoch模型
      model_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_{str(epoch)}.pt"
      dt_now = datetime.datetime.now()
      h_m_s = "{}:{}:{}".format(dt_now.strftime('%H'), dt_now.strftime('%M'),dt_now.strftime('%S'))
      train_info = f"| {h_m_s} | Fold: {fold_i} | Epoch: {'%03d'%epoch} | TraLoss: {'%.2f'%training_loss} | ValLoss |: {'%.2f'%validation_loss} | LR: {'%.6f'%self.scheduler._last_lr[0]}"
      print(train_info)
      train_info_list.append({'Fold':fold_i, 'Epoch':epoch, 'TrainLoss':training_loss, 'ValLoss':validation_loss})

      # 打印训练和验证结果
      train_ref, train_predict = result_process(train_results)
      test_ref, test_predict = result_process(test_results)
      train_eval = classification_eval(train_ref, train_predict)
      train_eval.pop('confusion_matrix')
      test_eval = classification_eval(test_ref, test_predict)
      test_eval.pop('confusion_matrix')
      print(f"Train: {train_eval}")
      print(f"Test: {test_eval}")

      # 保存最好的模型
      if validation_loss < best_valid_loss:
        best_valid_loss = validation_loss
        best_model_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_best.pt"
        save_checkpoint(self.model, model_path)
        save_checkpoint(self.model, best_model_path)

      # 早停机制
      if self.earlystop and epoch > 3:
        early_stopping(validation_loss)
        if early_stopping.early_stop:
          print("Early stopping")
          break

    # 训练过程可视化
    train_val_loss_df = pd.DataFrame.from_dict({'train':self.train_losses, 'val':self.val_losses}).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.savefig(f"{self.configs['model_dir']}/pics/{self.configs['model_name']}_{self.label}_fold{str(fold_i)}.png")
    plt.clf()

    # 保存训练数据
    info_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_loss.csv"
    df = pd.DataFrame(train_info_list)
    df.to_csv(info_path, index=False)

  def evaluate(self, test_loader, device):
    with torch.no_grad():
      keys, predictions, values = [], [], []
      for batch_idx, batch in enumerate(test_loader):
        key, audio_feats, video_feats, target, feats_lengths, video_len = batch
        audio_feats = audio_feats.to(device)
        video_feats = video_feats.to(device)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        video_len = video_len.to(device)
        self.model.eval()
        yhat = self.model(audio_feats, video_feats, feats_lengths)
        if self.task_type == 'classification':
          yhat = torch.argmax(yhat, dim=1)
        keys.extend(key)
        predictions.extend(yhat.cpu().detach().tolist())
        values.extend(target.unsqueeze(1).cpu().detach().tolist())
    return keys, predictions, values

  def get_middle_layer(self, test_loader, layername, device):
    with torch.no_grad():
      middle_list = []
      values = []
      relu_out = None
      for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        self.model.eval()
        model_wrapped = Inspect(self.model, layer=layername)
        y, x2 = model_wrapped(x_test)
        feat_label = torch.cat((x2.squeeze(), y.squeeze(0), y_test.unsqueeze(1)), dim=1) # 最后两维分别是预测值和目标值
        middle_list.append(feat_label.cpu().detach().numpy())
    middle_data = np.concatenate(middle_list, axis=0)
    column_names = ['feat_{}'.format(str(i)) for i in range(middle_data.shape[1]-2)]
    column_names.append('Predict')
    column_names.append('Target')
    df_middle_data = pd.DataFrame(middle_data, columns=column_names)
    return df_middle_data
  

class MTLAddTrainer(BaseTrainer):
  def __init__(self, configs, label, model, loss_fn, optimizer, scheduler):
    super().__init__(configs, label, model, loss_fn, optimizer, scheduler)
    self.accum_grad = 1 if 'accum_grad' not in self.configs else self.configs['accum_grad']
    # 分类还是回归
    self.task_type = self.configs['task_info']['type']
    self.earlystop = self.configs['earlystop']
    self.n_epochs = self.configs['EPOCHS']
    self.batch_size = self.configs['batch_size']
    self.loss_fusion = self.configs.get('loss_fusion', 'add')
    if self.loss_fusion == 'auto':
      self.MultiLoss = MultiTaskLoss()
      self.optimizer.add_param_group({'params': self.MultiLoss.parameters()})
    self.clip_grad = self.configs.get('clip_grad', 0)

  def train_step(self, audio_feats, video_feats, feats_lens, target, ):
    self.model.train()
    fusion_out, audio_out, visual_out = self.model(audio_feats, video_feats, feats_lens, target)
    if self.task_type == 'classification':
      # loss = self.loss_fn(yhat, target.to(torch.int64))
      fusion_loss = self.loss_fn(fusion_out, target.squeeze(1).to(torch.int64))
      audio_loss = self.loss_fn(audio_out, target.squeeze(1).to(torch.int64))
      video_loss = self.loss_fn(visual_out, target.squeeze(1).to(torch.int64))
    elif self.task_type == 'regression':
      fusion_loss = self.loss_fn(fusion_out, target.squeeze(1))
      audio_loss = self.loss_fn(audio_out, target.squeeze(1))
      video_loss = self.loss_fn(visual_out, target.squeeze(1))
    if self.loss_fusion == 'add':
      loss = fusion_loss + audio_loss + video_loss
    elif self.loss_fusion == 'auto':
      loss = self.MultiLoss([fusion_loss, audio_loss, video_loss])
    elif self.loss_fusion == 'a+v':
      loss = audio_loss + video_loss

    loss = loss/self.accum_grad
    loss.backward()
    if self.clip_grad > 0:
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
      if self.loss_fusion == 'auto':
        torch.nn.utils.clip_grad_norm_(self.MultiLoss.parameters(), self.clip_grad)
    return loss.item()

  def train(self, train_loader, val_loader, device, fold_i):
    if self.earlystop > 0:
      early_stopping = EarlyStopping(patience=self.earlystop, verbose=True)
    train_info_list = []
    best_valid_loss = 10000

    for epoch in range(1, self.n_epochs + 1):
      batch_losses = []
      # 训练
      for batch_idx, batch in enumerate(train_loader):
        key, audio_feats, video_feats, target, feats_lens = batch
        audio_feats = audio_feats.to(device)
        video_feats = video_feats.to(device)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        if audio_feats.size(0) != self.batch_size:
          print(f"batch {batch_idx} size not match")
          continue

        loss = self.train_step(audio_feats, video_feats, feats_lens, target)

        if ((batch_idx) % self.accum_grad == 0):
          self.optimizer.step()
          self.optimizer.zero_grad()
        if np.isnan(loss):
          print('nan loss')
        else:
          batch_losses.append(loss)
      self.scheduler.step()
      training_loss = np.mean(batch_losses)
      self.train_losses.append(training_loss*self.accum_grad)

      # 验证
      with torch.no_grad():
        batch_val_losses = []

        for batch_idx, batch in enumerate(val_loader):
          key, audio_feats, video_feats, target, feats_lens = batch
          audio_feats = audio_feats.to(device)
          video_feats = video_feats.to(device)
          target = target.to(device)
          feats_lengths = feats_lengths.to(device)
          if audio_feats.size(0) ==0 or video_feats.size(0) == 0 or target.size(0)==0:
            print(f"batch {batch_idx} size not match")
            continue
          self.model.eval()
          fusion_out, audio_out, visual_out = self.model(audio_feats, video_feats, feats_lens, target)
          if self.task_type == 'classification':
            # val_loss = self.loss_fn(yhat, target.to(torch.int64)).item()
            fusion_loss = self.loss_fn(fusion_out, target.squeeze(1).to(torch.int64))
            audio_loss = self.loss_fn(audio_out, target.squeeze(1).to(torch.int64))
            video_loss = self.loss_fn(visual_out, target.squeeze(1).to(torch.int64))
            val_loss = fusion_loss + audio_loss + video_loss
          elif self.task_type == 'regression':
            fusion_loss = self.loss_fn(fusion_out, target.squeeze(1))
            audio_loss = self.loss_fn(audio_out, target.squeeze(1))
            video_loss = self.loss_fn(visual_out, target.squeeze(1))
            val_loss = fusion_loss + audio_loss + video_loss
          val_loss = val_loss.cpu().numpy()
          if np.isnan(val_loss):
            print('nan loss')
          else:
            batch_val_losses.append(val_loss)
        validation_loss = np.mean(batch_val_losses)
        self.val_losses.append(validation_loss)

      # 保存每个epoch模型
      model_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_{str(epoch)}.pt"
      dt_now = datetime.datetime.now()
      h_m_s = "{}:{}:{}".format(dt_now.strftime('%H'), dt_now.strftime('%M'),dt_now.strftime('%S'))
      train_info = f"| {h_m_s} | Fold: {fold_i} | Epoch: {'%03d'%epoch} | TraLoss: {'%.2f'%training_loss} | ValLoss |: {'%.2f'%validation_loss} | LR: {'%.6f'%self.scheduler._last_lr[0]}"
      print(train_info)
      train_info_list.append({'Fold':fold_i, 'Epoch':epoch, 'TrainLoss':training_loss, 'ValLoss':validation_loss})

      # 保存最好的模型
      if validation_loss < best_valid_loss:
        best_valid_loss = validation_loss
        best_model_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_best.pt"
        save_checkpoint(self.model, model_path)
        save_checkpoint(self.model, best_model_path)

      # 早停机制，为避免过早停，在10轮之后计数
      if self.earlystop and epoch > 10*self.accum_grad:
        early_stopping(validation_loss)
        if early_stopping.early_stop:
          print("Early stopping")
          break

    # 训练过程可视化
    train_val_loss_df = pd.DataFrame.from_dict({'train':self.train_losses, 'val':self.val_losses}).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    # plt.figure(figsize=(15,8))
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.savefig(f"{self.configs['model_dir']}/pics/{self.configs['model_name']}_{self.label}_fold{str(fold_i)}.png")
    plt.clf()

    # 保存训练数据
    info_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_loss.csv"
    df = pd.DataFrame(train_info_list)
    df.to_csv(info_path, index=False)

  def evaluate(self, test_loader, device):
    with torch.no_grad():
      keys, predictions, values = [], [], []
      for batch_idx, batch in enumerate(test_loader):
        key, audio_feats, video_feats, target, feats_lens = batch
        audio_feats = audio_feats.to(device)
        video_feats = video_feats.to(device)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        self.model.eval()
        fusion_out, audio_out, visual_out = self.model(audio_feats, video_feats, feats_lens, target)
        if self.task_type == 'classification':
          fusion_out = torch.argmax(fusion_out, dim=1)
        keys.extend(key)
        predictions.extend(fusion_out.cpu().detach().tolist())
        values.extend(target.unsqueeze(1).cpu().detach().tolist())
    return keys, predictions, values

  def get_middle_layer(self, test_loader, layername, device):
    with torch.no_grad():
      middle_list = []
      values = []
      relu_out = None
      for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        self.model.eval()
        model_wrapped = Inspect(self.model, layer=layername)
        y, x2 = model_wrapped(x_test)
        feat_label = torch.cat((x2.squeeze(), y.squeeze(0), y_test.unsqueeze(1)), dim=1) # 最后两维分别是预测值和目标值
        middle_list.append(feat_label.cpu().detach().numpy())
    middle_data = np.concatenate(middle_list, axis=0)
    column_names = ['feat_{}'.format(str(i)) for i in range(middle_data.shape[1]-2)]
    column_names.append('Predict')
    column_names.append('Target')
    df_middle_data = pd.DataFrame(middle_data, columns=column_names)
    return df_middle_data
  

class MTLMMPTrainer(BaseTrainer):
  def __init__(self, configs, label, model, loss_fn, optimizer, scheduler):
    super().__init__(configs, label, model, loss_fn, optimizer, scheduler)
    self.accum_grad = 1 if 'accum_grad' not in self.configs else self.configs['accum_grad']
    # 分类还是回归
    self.task_type = self.configs['task_info']['type']
    self.earlystop = self.configs['earlystop']
    self.n_epochs = self.configs['EPOCHS']
    self.batch_size = self.configs['batch_size']
    self.clip_grad = self.configs.get('clip_grad', 0)


  def train(self, train_loader, val_loader, device, fold_i):
    if self.earlystop > 0:
      early_stopping = EarlyStopping(patience=self.earlystop, verbose=True)
    train_info_list = []
    best_valid_loss = 10000

    for epoch in range(1, self.n_epochs + 1):
      batch_losses = []
      self.model.train()
      print("Start training ... ")
      loss_value_mm=[]
      loss_value_a=[]
      loss_value_v=[]

      record_names_audio = []
      record_names_visual = []
      for name, param in self.model.named_parameters():
        if 'head' in name: 
          continue
        if ('audio' in name):
          record_names_audio.append((name, param))
          continue
        if ('visual' in name):
          record_names_visual.append((name, param))
          continue
      
      # 训练
      for batch_idx, batch in enumerate(train_loader):
        key, audio_feats, video_feats, target, feats_lens = batch
        audio_feats = audio_feats.to(device)
        video_feats = video_feats.to(device)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        # if audio_feats.size(0) ==0 or video_feats.size(0) == 0 or target.size(0)==0:
        if audio_feats.size(0) != self.batch_size:
          print(f"batch {batch_idx} size not match")
          continue
        
        self.optimizer.zero_grad()
        fusion_out, audio_out, visual_out = self.model(audio_feats, video_feats, feats_lens, target)
        loss_mm = self.loss_fn(fusion_out, target.squeeze(1).to(torch.int64))
        loss_a = self.loss_fn(audio_out, target.squeeze(1).to(torch.int64))
        loss_v = self.loss_fn(visual_out, target.squeeze(1).to(torch.int64))
        loss_value_mm.append(loss_mm.item())
        loss_value_a.append(loss_a.item())
        loss_value_v.append(loss_v.item())

        losses = [loss_mm, loss_a, loss_v]
        all_loss = ['both', 'audio', 'visual']

        grads_audio = {}
        grads_visual = {}

        for idx, loss_type in enumerate(all_loss):
          loss = losses[idx]
          loss.backward(retain_graph=True)

          if(loss_type=='visual'):
            for tensor_name, param in record_names_visual:
              if loss_type not in grads_visual.keys():
                grads_visual[loss_type] = {}
              if param.grad is not None:
                grads_visual[loss_type][tensor_name] = param.grad.data.clone()
            visual_para_grads = []
            for tensor_name, param in record_names_visual:
              if tensor_name in grads_visual[loss_type].keys():
                visual_para_grads.append(grads_visual[loss_type][tensor_name].flatten())
            grads_visual[loss_type]["concat"] = torch.cat(visual_para_grads)
          elif(loss_type=='audio'):
            for tensor_name, param in record_names_audio:
              if loss_type not in grads_audio.keys():
                grads_audio[loss_type] = {}
              grads_audio[loss_type][tensor_name] = param.grad.data.clone()
            grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_audio])
          else:
            for tensor_name, param in record_names_audio:
              if loss_type not in grads_audio.keys():
                grads_audio[loss_type] = {}
              grads_audio[loss_type][tensor_name] = param.grad.data.clone() 
            grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
            for tensor_name, param in record_names_visual:
              if loss_type not in grads_visual.keys():
                grads_visual[loss_type] = {}
              if param.grad is not None:
                grads_visual[loss_type][tensor_name] = param.grad.data.clone()
            
            visual_para_grads = []
            for tensor_name, param in record_names_visual:
              if tensor_name in grads_visual[loss_type].keys():
                visual_para_grads.append(grads_visual[loss_type][tensor_name].flatten())
            grads_visual[loss_type]["concat"] = torch.cat(visual_para_grads)

          self.optimizer.zero_grad()
        this_cos_audio=F.cosine_similarity(grads_audio['both']["concat"],grads_audio['audio']["concat"],dim=0)
        this_cos_visual=F.cosine_similarity(grads_visual['both']["concat"],grads_visual['visual']["concat"],dim=0)
        audio_task=['both','audio']
        visual_task=['both','visual']

        audio_k=[0,0]
        visual_k=[0,0]
        if(this_cos_audio>0):
          audio_k[0]=0.5
          audio_k[1]=0.5
        else:
          audio_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_audio[t].values()) for t in audio_task])
        if(this_cos_visual>0):
          visual_k[0]=0.5
          visual_k[1]=0.5
        else:
          visual_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_visual[t].values()) for t in visual_task])
        gamma=1.5
        loss=loss_mm+loss_a+loss_v
        loss.backward()

        if self.clip_grad > 0:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)


        for name, param in self.model.named_parameters():
          if param.grad is not None:
            layer = re.split('[_.]',str(name))
            if('head' in layer):
              continue
            if('audio' in layer):
              three_norm=torch.norm(param.grad.data.clone())
              new_grad=2*audio_k[0]*grads_audio['both'][name]+2*audio_k[1]*grads_audio['audio'][name]
              new_norm=torch.norm(new_grad)
              diff=three_norm/new_norm
              if(diff>1):
                param.grad=diff*new_grad*gamma
              else:
                param.grad=new_grad*gamma
            if('visual' in layer):
              three_norm=torch.norm(param.grad.data.clone())
              new_grad=2*visual_k[0]*grads_visual['both'][name]+2*visual_k[1]*grads_visual['visual'][name]
              new_norm=torch.norm(new_grad)
              diff=three_norm/new_norm
              if(diff>1):
                param.grad=diff*new_grad*gamma
              else:
                param.grad=new_grad*gamma

        loss = loss.item()
        if ((batch_idx) % self.accum_grad == 0):
          self.optimizer.step()
          self.optimizer.zero_grad()
        if np.isnan(loss):
          print('nan loss')
        else:
          batch_losses.append(loss)
        print('train_loss', loss)
      
      self.scheduler.step()
      training_loss = np.mean(batch_losses)
      self.train_losses.append(training_loss*self.accum_grad)

      # 验证
      with torch.no_grad():
        batch_val_losses = []
        for batch_idx, batch in enumerate(val_loader):
          key, audio_feats, video_feats, target, feats_lens = batch
          audio_feats = audio_feats.to(device)
          video_feats = video_feats.to(device)
          target = target.to(device)
          feats_lengths = feats_lengths.to(device)
          if audio_feats.size(0) ==0 or video_feats.size(0) == 0 or target.size(0)==0:
            print(f"batch {batch_idx} size not match")
            continue
          self.model.eval()
          fusion_out, audio_out, visual_out = self.model(audio_feats, video_feats, feats_lens, target)
          if self.task_type == 'classification':
            # val_loss = self.loss_fn(yhat, target.to(torch.int64)).item()
            fusion_loss = self.loss_fn(fusion_out, target.squeeze(1).to(torch.int64))
            audio_loss = self.loss_fn(audio_out, target.squeeze(1).to(torch.int64))
            visual_loss = self.loss_fn(visual_out, target.squeeze(1).to(torch.int64))
            val_loss = fusion_loss + audio_loss + visual_loss
          elif self.task_type == 'regression':
            fusion_loss = self.loss_fn(fusion_out, target.squeeze(1)).item()
            audio_loss = self.loss_fn(audio_out, target.squeeze(1)).item()
            visual_loss = self.loss_fn(visual_out, target.squeeze(1)).item()
            val_loss = fusion_loss + audio_loss + visual_loss
          val_loss = val_loss.cpu().numpy()
          if np.isnan(val_loss):
            print('nan loss')
          else:
            batch_val_losses.append(val_loss)
        validation_loss = np.mean(batch_val_losses)
        self.val_losses.append(validation_loss)

      # 保存每个epoch模型
      model_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_{str(epoch)}.pt"
      dt_now = datetime.datetime.now()
      h_m_s = "{}:{}:{}".format(dt_now.strftime('%H'), dt_now.strftime('%M'),dt_now.strftime('%S'))
      train_info = f"| {h_m_s} | Fold: {fold_i} | Epoch: {'%03d'%epoch} | TraLoss: {'%.2f'%training_loss} | ValLoss |: {'%.2f'%validation_loss} | LR: {'%.6f'%self.scheduler._last_lr[0]}"
      print(train_info)
      train_info_list.append({'Fold':fold_i, 'Epoch':epoch, 'TrainLoss':training_loss, 'ValLoss':validation_loss})

      # 保存最好的模型
      if validation_loss < best_valid_loss:
        best_valid_loss = validation_loss
        best_model_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_best.pt"
        save_checkpoint(self.model, model_path)
        save_checkpoint(self.model, best_model_path)

      # 早停机制，为避免过早停，在10轮之后计数
      if self.earlystop and epoch > 10*self.accum_grad:
        early_stopping(validation_loss)
        if early_stopping.early_stop:
          print("Early stopping")
          break

    # 训练过程可视化
    train_val_loss_df = pd.DataFrame.from_dict({'train':self.train_losses, 'val':self.val_losses}).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    # plt.figure(figsize=(15,8))
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.savefig(f"{self.configs['model_dir']}/pics/{self.configs['model_name']}_{self.label}_fold{str(fold_i)}.png")
    plt.clf()

    # 保存训练数据
    info_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_loss.csv"
    df = pd.DataFrame(train_info_list)
    df.to_csv(info_path, index=False)

  def evaluate(self, test_loader, device):
    with torch.no_grad():
      keys, predictions, values = [], [], []
      for batch_idx, batch in enumerate(test_loader):
        key, audio_feats, video_feats, target, feats_lens = batch
        audio_feats = audio_feats.to(device)
        video_feats = video_feats.to(device)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        self.model.eval()
        fusion_out, audio_out, visual_out = self.model(audio_feats, video_feats, feats_lens, target)
        if self.task_type == 'classification':
          fusion_out = torch.argmax(fusion_out, dim=1)
          audio_out = torch.argmax(audio_out, dim=1)
          visual_out = torch.argmax(visual_out, dim=1)
        keys.extend(key)
        predictions.extend(fusion_out.cpu().detach().tolist())
        values.extend(target.unsqueeze(1).cpu().detach().tolist())
    return keys, predictions, values

  def get_middle_layer(self, test_loader, layername, device):
    with torch.no_grad():
      middle_list = []
      values = []
      relu_out = None
      for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        self.model.eval()
        model_wrapped = Inspect(self.model, layer=layername)
        y, x2 = model_wrapped(x_test)
        feat_label = torch.cat((x2.squeeze(), y.squeeze(0), y_test.unsqueeze(1)), dim=1) # 最后两维分别是预测值和目标值
        middle_list.append(feat_label.cpu().detach().numpy())
    middle_data = np.concatenate(middle_list, axis=0)
    column_names = ['feat_{}'.format(str(i)) for i in range(middle_data.shape[1]-2)]
    column_names.append('Predict')
    column_names.append('Target')
    df_middle_data = pd.DataFrame(middle_data, columns=column_names)
    return df_middle_data
  
