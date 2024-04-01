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

import torch
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from networks.utils.basetrainer import BaseTrainer
from networks.utils.checkpoint import save_checkpoint
from networks.utils.early_stop import EarlyStopping
from surgeon_pytorch import Inspect, get_layers


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
          batch_val_losses.append(val_loss)
        validation_loss = np.mean(batch_val_losses)
        self.val_losses.append(validation_loss)

      # 保存每个epoch模型
      if ((epoch) % self.accum_grad == 0) or (epoch == self.n_epochs):
        model_path = f"{self.configs['model_dir']}/models/{self.label}_fold_{str(fold_i)}_{str(epoch)}.pt"
        dt_now = datetime.datetime.now()
        h_m_s = "{}:{}:{}".format(dt_now.strftime('%H'), dt_now.strftime('%M'),dt_now.strftime('%S'))
        train_info = f"| {h_m_s} | Fold: {fold_i} | Epoch: {'%03d'%epoch} | TraLoss: {'%.2f'%training_loss} | ValLoss |: {'%.2f'%validation_loss} | LR: {'%.4f'%self.scheduler._last_lr[0]}"
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
