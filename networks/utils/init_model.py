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
import os
import torch

from networks.model.lstm import LSTMNet 

from networks.model.video.tsn import TSN 
from networks.model.video.twostream import TWOSTREAM
from networks.model.video.tsn_seg import TSNSEG
from networks.model.video.tsn_split import TSNSPLIT
from networks.model.video.tsn_seg_split import TSNSEGSPLIT
from networks.model.video.tsn_tseg_ssplit import TSNSSEGTSPLIT
from networks.model.video.swin_transformer import SwinTransformer
from networks.model.video.resnet_3d import generate_model

from networks.model.transformer.encoder import ConformerEncoder
from networks.model.transformer.enc_linear import EncLinear
from networks.model.transformer.enc_lstm import EncLSTM
from networks.model.transformer.Wav2vecNet import Wav2vecNet
from networks.model.squeezeformer.encoder import SqueezeformerEncoder
from networks.model.zipformer.encoder import Zipformer2
from networks.model.zipformer.zip_linear import ZipLinear
from networks.model.audio_resnet.resnet import get_resnet
from networks.model.audio_resnet.resnetsqu import get_resnetfus, get_resnetsqu
from networks.model.audio_resnet.res2net import get_res2net
from networks.model.audio_resnet.resnet_seg import get_resnet10seg
from networks.model.audio_resnet.se_resnet import get_seresnet

from networks.model.linear import LinearNet, LinearNet_1
from networks.model.audio_visual.fusion_model import MultiLoss
from networks.model.audio_visual.corss_fusion_model import CrossAVFusion, ShareAVFusion, CatAVFusion, ShareCrossAVFusion
from networks.model.audio_visual.avcnn import AudioVideoCNN, AudioCNN, VideoCNN
from networks.model.audio_visual.avhubert import AVHubertLSTM, AHubertLSTM, AVHubertMLP, AHubertMLP, VHubertMLP
from networks.model.text_distance.text_MLP import MLPNet, DistanceNet, AttensionNet

from networks.utils.checkpoint import load_checkpoint, load_trained_modules

def migration(model, configs, fold_i):
    if 'encoder_flag' in configs and configs['encoder_flag'] is not None:
        encoder_flag = configs['encoder_flag']
    else:
        encoder_flag = 0
    if 'checkpoint' in configs and configs['checkpoint'] is not None:
      checkpoint = configs['checkpoint'].replace('fold_0', f'fold_{fold_i}')
      if os.path.exists(checkpoint):
        infos = load_checkpoint(model, checkpoint)
        print(f'Load checkpoint: {checkpoint}')
      else:
        infos = {}
        print(f'No such checkpoint: {checkpoint}')
    elif 'enc_init' in configs and configs['enc_init'] is not None:
        pretrain_model = configs['enc_init'].replace('fold_0', f'fold_{fold_i}')
        infos = load_trained_modules(model, pretrain_model, encoder_flag)
        print(f'Load pretrain model: {pretrain_model}')
    else:
        print('No checkpoint or pretrain model')
        infos = {}
    configs["init_infos"] = infos
    print(configs)
    return model

# init model
def init_model(configs, fold):
    model_type = configs.get('model_name')
    if model_type == 'lstm':
        model = LSTMNet(
            input_dim=configs['input_dim'],
            hidden_dim=configs['hidden_dim'],
            output_dim=configs['task_info']['score'])
    elif model_type == 'resnet3d':
        encoder = generate_model(configs['model_depth'], out_dim=configs['resnet3d_out_dim'])
        model = EncLinear(configs['model_name'], encoder, configs['task_info']['score'], configs['dropout'])
    elif model_type == 'tsn':
        model = TSN(
            out_dim=configs['task_info']['score'],
            num_segments=configs['num_segments'],
            modality=configs['modality'],
            new_length=configs['new_length'],
            dropout=configs['dropout'])
    elif model_type == 'tsn_split':
        model = TSNSPLIT(
            out_dim=configs['task_info']['score'],
            num_segments=configs['num_segments'],
            modality=configs['modality'],
            new_length=configs['new_length'],
            dropout=configs['dropout'])
    elif model_type == "swin_transformer":
        model = SwinTransformer(
            patch_size=configs['patch_size'],
            num_classes=configs['out_dim'],
            window_size=configs['window_size'],
            drop_rate=configs['drop_rate'],
            num_segments=configs['num_segments'])
    elif model_type == 'twostream':
        model = TWOSTREAM(
            out_dim=configs['out_dim'],
            num_class=configs['task_info']['score'],
            num_segments=configs['num_segments'],
            modality=configs['modality'],
            dropout=configs['dropout'])
    elif model_type == 'tsn_seg':
        model = TSNSEG(
            num_class=configs['task_info']['score'],
            out_dim=configs['out_dim'],
            num_segments=configs['num_segments'],
            modality=configs['modality'],
            new_length=configs['new_length'],
            consensus_type=configs['consensus_type'],
            dropout=configs['dropout'])
    elif model_type == 'tsn_segsplit':
        model = TSNSEGSPLIT(
            num_class=configs['task_info']['score'],
            out_dim=configs['out_dim'],
            num_segments=configs['num_segments'],
            modality=configs['modality'],
            new_length=configs['new_length'],
            consensus_type=configs['consensus_type'],
            window_size=configs['window_size'],
            dropout=configs['dropout'])
    elif model_type == 'conformer6linear' or model_type == 'conformer3linear' or model_type == 'conformer3slinear':
        encoder = ConformerEncoder(configs['input_dim'],
                                   global_cmvn=None,
                                   **configs['encoder_conf'])
        model = EncLinear(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'conformer3lstm' or model_type == 'conformer6lstm':
        encoder = ConformerEncoder(configs['input_dim'],  global_cmvn=None, **configs['encoder_conf'])
        model = EncLSTM(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'squeezeformer6linear':
        encoder = SqueezeformerEncoder(configs['input_dim'],
                                        global_cmvn=None,
                                        **configs['encoder_conf'])
        model = EncLinear(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'zip_linear':
        encoder_conf = configs['encoder_conf']
        zipformer2 = Zipformer2(encoder_dim=encoder_conf['encoder_dim'], feature_dim=configs['input_dim'])
        model = ZipLinear(encoder_dim=encoder_conf['encoder_dim'], encoder=zipformer2, vocab_size=configs['output_dim'])
    elif model_type == 'resnet34linear' or model_type == 'resnet18linear' or model_type == 'resnet10linear':
        encoder_conf = configs['encoder_conf']
        encoder = get_resnet(resnum=encoder_conf['resnum'], feat_dim=configs['input_dim'], embed_dim=encoder_conf['embed_dim'])
        model = EncLinear(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'resnet34lstm' or model_type == 'resnet18lstm' or model_type == 'resnet10lstm':
        encoder_conf = configs['encoder_conf']
        encoder = get_resnet(resnum=encoder_conf['resnum'], feat_dim=configs['input_dim'], embed_dim=encoder_conf['embed_dim'])
        model = EncLSTM(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'resnetseq34linear' or model_type == 'resnetseq18linear' or model_type == 'resnetseq10linear':
        encoder_conf = configs['encoder_conf']
        encoder = get_resnetsqu(resnum=encoder_conf['resnum'], feat_dim=configs['input_dim'], embed_dim=encoder_conf['embed_dim'])
        model = EncLinear(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif 'resnetfus' in model_type:
        encoder_conf = configs['encoder_conf']
        encoder = get_resnetfus(resnum=encoder_conf['resnum'], feat_dim=configs['input_dim'], embed_dim=encoder_conf['embed_dim'])
        model = EncLinear(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'res2net10linear' or model_type == 'res2net18linear' or model_type == 'res2net34linear':
        encoder_conf = configs['encoder_conf']
        encoder = get_res2net(resnum=encoder_conf['resnum'], feat_dim=configs['input_dim'], embed_dim=encoder_conf['embed_dim'])
        model = EncLinear(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'resnet10seg2' or model_type == 'resnet10seg3' or model_type == 'resnet10seg4' or model_type == 'resnet10seg5':
        encoder_conf = configs['encoder_conf']
        segnum = configs['audiosegnum']
        encoder = get_resnet10seg(segnum, encoder_conf['embed_dim'], base_model=encoder_conf['model_type'], resnum=encoder_conf['resnum'], feat_dim=encoder_conf['feat_dim'])
        model = EncLinear(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'seresnet10linear' or model_type == 'seresnet18linear' or model_type == 'seresnet50linear':
        encoder_conf = configs['encoder_conf']
        segnum = encoder_conf['resnum']
        encoder = get_seresnet(resnum=encoder_conf['resnum'], embed_dim=encoder_conf['embed_dim'],  feat_dim=encoder_conf['input_dim'])
        model = EncLinear(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'goplstmlinear' or model_type == 'wav2veclstm':
        encoder_conf = configs['encoder_conf']
        encoder = LSTMNet(**encoder_conf)
        model = EncLinear(configs['model_name'], encoder, configs['output_dim'], configs['dropout'])
    elif model_type == 'goplinear' or model_type == 'wav2veclinear' or model_type == 'wav2vecmeanlinear':
        encoder_conf = configs['encoder_conf']
        model = LinearNet(**encoder_conf)
    elif model_type == 'ivectormlp' or model_type == 'egemapsmlp':
        encoder_conf = configs['encoder_conf']
        model = LinearNet_1(**encoder_conf)
    elif model_type == 'wav2vec_mlp':
        encoder_conf = configs['encoder_conf']
        model = Wav2vecNet(**encoder_conf)
    elif model_type.split('_')[0] == 'av':
        audio_conf = configs['audio_conf']
        visual_conf = configs['visual_conf']
        audio_model = get_resnetsqu(resnum=audio_conf['resnum'], feat_dim=audio_conf['input_dim'], embed_dim=audio_conf['embed_dim'])
        audio_model = migration(audio_model, audio_conf, fold)
        visual_model = TSNSEGSPLIT(
            num_class=configs['task_info']['score'],
            out_dim=visual_conf['out_dim'],
            num_segments=visual_conf['num_segments'],
            modality=visual_conf['modality'],
            new_length=visual_conf['new_length'],
            consensus_type=visual_conf['consensus_type'],
            window_size=visual_conf['window_size'],
            dropout=visual_conf['dropout'])
        visual_model = migration(visual_model, visual_conf, fold)
        model = MultiLoss(fold, audio_model, visual_model, audio_conf['embed_dim'], visual_conf['out_dim'], configs['av_fusion_type'], configs['output_dim'], configs['dropout'])
    elif 'crossshareav_resnetseq10_tsnsegsplit' in model_type.lower():
        audio_conf = configs['audio_conf']
        visual_conf = configs['visual_conf']
        audio_model = get_resnetsqu(resnum=audio_conf['resnum'], feat_dim=audio_conf['input_dim'], embed_dim=audio_conf['embed_dim'])
        audio_model = migration(audio_model, audio_conf, fold)
        visual_model = TSNSEGSPLIT(
            num_class=configs['task_info']['score'],
            out_dim=visual_conf['out_dim'],
            num_segments=visual_conf['num_segments'],
            modality=visual_conf['modality'],
            new_length=visual_conf['new_length'],
            consensus_type=visual_conf['consensus_type'],
            window_size=visual_conf['window_size'],
            dropout=visual_conf['dropout'])
        visual_model = migration(visual_model, visual_conf, fold)
        model = ShareCrossAVFusion(fold, audio_model, visual_model, audio_conf['embed_dim'], visual_conf['out_dim'], configs['output_dim'], configs['av_fusion_type'], configs['av_corss_type'], configs['att_type'], configs['similarity_type'], configs['dropout'])
    elif 'crossav_resnetseq10_tsnsegsplit' in model_type.lower():
        audio_conf = configs['audio_conf']
        visual_conf = configs['visual_conf']
        audio_model = get_resnetsqu(resnum=audio_conf['resnum'], feat_dim=audio_conf['input_dim'], embed_dim=audio_conf['embed_dim'])
        audio_model = migration(audio_model, audio_conf, fold)
        visual_model = TSNSEGSPLIT(
            num_class=configs['task_info']['score'],
            out_dim=visual_conf['out_dim'],
            num_segments=visual_conf['num_segments'],
            modality=visual_conf['modality'],
            new_length=visual_conf['new_length'],
            consensus_type=visual_conf['consensus_type'],
            window_size=visual_conf['window_size'],
            dropout=visual_conf['dropout'])
        visual_model = migration(visual_model, visual_conf, fold)
        model = CrossAVFusion(fold, audio_model, visual_model, audio_conf['embed_dim'], visual_conf['out_dim'], configs['output_dim'], configs['av_fusion_type'], configs['av_corss_type'], configs['att_type'], configs['similarity_type'], configs['dropout'])
    elif 'shareav_resnetseq10_tsnsegsplit' in model_type.lower():
        audio_conf = configs['audio_conf']
        visual_conf = configs['visual_conf']
        audio_model = get_resnetsqu(resnum=audio_conf['resnum'], feat_dim=audio_conf['input_dim'], embed_dim=audio_conf['embed_dim'])
        audio_model = migration(audio_model, audio_conf, fold)
        visual_model = TSNSEGSPLIT(
            num_class=configs['task_info']['score'],
            out_dim=visual_conf['out_dim'],
            num_segments=visual_conf['num_segments'],
            modality=visual_conf['modality'],
            new_length=visual_conf['new_length'],
            consensus_type=visual_conf['consensus_type'],
            window_size=visual_conf['window_size'],
            dropout=visual_conf['dropout'])
        visual_model = migration(visual_model, visual_conf, fold)
        model = ShareAVFusion(fold, audio_model, visual_model, audio_conf['embed_dim'], visual_conf['out_dim'], configs['output_dim'], configs['av_fusion_type'], configs['av_corss_type'], configs['att_type'], configs['similarity_type'], configs['dropout'])
    elif model_type.split('_')[0] == 'concatav':
        audio_conf = configs['audio_conf']
        visual_conf = configs['visual_conf']
        audio_model = get_resnetsqu(resnum=audio_conf['resnum'], feat_dim=audio_conf['input_dim'], embed_dim=audio_conf['embed_dim'])
        audio_model = migration(audio_model, audio_conf, fold)
        visual_model = TSNSEGSPLIT(
            num_class=configs['task_info']['score'],
            out_dim=visual_conf['out_dim'],
            num_segments=visual_conf['num_segments'],
            modality=visual_conf['modality'],
            new_length=visual_conf['new_length'],
            consensus_type=visual_conf['consensus_type'],
            window_size=visual_conf['window_size'],
            dropout=visual_conf['dropout'])
        visual_model = migration(visual_model, visual_conf, fold)
        model = CatAVFusion(fold, audio_model, visual_model, audio_conf['embed_dim'], visual_conf['out_dim'], configs['av_fusion_type'], configs['output_dim'])
    elif model_type == 'avcnn':
        model = AudioVideoCNN(configs['audio_conf']['out_dim'], configs['visual_conf']['out_dim'], configs['output_dim'])
    elif model_type == 'avcnn_a':
        model = AudioCNN(configs['audio_conf']['out_dim'], configs['output_dim'])
    elif model_type == 'avcnn_v':
        model = VideoCNN(configs['visual_conf']['out_dim'], configs['output_dim'])
    elif model_type == 'avhubert_lstm':
        model = AVHubertLSTM(configs['audio_conf']['input_dim'], configs['audio_conf']['input_dim'], configs['audio_conf']['out_dim'], configs['visual_conf']['out_dim'], configs['output_dim'])
    elif model_type == 'ahubert_lstm':
        model = AHubertLSTM(configs['audio_conf']['input_dim'], configs['audio_conf']['out_dim'], configs['output_dim'])
    elif model_type == 'avhubert_mlp':
        model = AVHubertMLP(configs['audio_conf']['input_dim'], configs['audio_conf']['input_dim'], configs['audio_conf']['out_dim'], configs['visual_conf']['out_dim'], configs['output_dim'])
    elif model_type == 'ahubert_mlp' or model_type == 'vhubert_mlp':
        if model_type == 'ahubert_mlp':
            model = AHubertMLP(configs['audio_conf']['input_dim'], configs['audio_conf']['out_dim'], configs['output_dim'])
        elif model_type == 'vhubert_mlp':
            model = VHubertMLP(configs['visual_conf']['input_dim'], configs['visual_conf']['out_dim'], configs['output_dim'])
    elif model_type == 'crossattmlp':
        encoder_config = configs['encoder_conf']
        model = MLPNet(encoder_config['input_dim'], encoder_config['hidden_dim'], encoder_config['output_dim'], encoder_config['droprate'])
    elif model_type == 'decoderdist_textdist_dis' or model_type == 'crossatt_textdist_dis':
        encoder_config = configs['encoder_conf']
        model = DistanceNet(encoder_config['input_dim'], encoder_config['hidden_dim'], encoder_config['output_dim'], encoder_config['droprate'])
    elif model_type == 'crossatt_textdist_att' or model_type == 'decoderdist_textdist_att':
        encoder_config = configs['encoder_conf']
        model = AttensionNet(encoder_config['input_dim'], encoder_config['hidden_dim'], encoder_config['output_dim'], encoder_config['droprate'])
    else:
        raise ValueError(f"model type {model_type} not supported")

    if 'encoder_flag' in configs and configs['encoder_flag'] is not None:
        encoder_flag = configs['encoder_flag']
    else:
        encoder_flag = 0
    # If specify checkpoint, load some info from checkpoint
    if 'checkpoint' in configs and configs['checkpoint'] is not None:
        checkpoint = configs['checkpoint'].replace('fold_0', f'fold_{fold}')
        infos = load_checkpoint(model, checkpoint)
    elif 'enc_init' in configs and configs['enc_init'] is not None:
        pretrain_model = configs['enc_init'].replace('fold_0', f'fold_{fold}')
        infos = load_trained_modules(model, pretrain_model, encoder_flag)
    else:
        infos = {}
    configs["init_infos"] = infos
    print(configs)
    return model, configs

