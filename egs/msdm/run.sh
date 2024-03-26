#!/bin/bash

. ./path.sh || exit 1;

stage=1 # start from 0 if you need to start from data preparation
stop_stage=1

# data preperation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0"
fi

# classfication
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1"
  # conf_file=vhubert_lstm_drop0.2.yaml
  conf_file=clmrv_lstm_drop0.2.yaml
  for fold in 0 1 2 3 4 5 6 7 8 9; do
    gpu_id=$((fold % 4))
    python local/main.py --config conf/$conf_file --gpu ${gpu_id} --fold ${fold} > log/${conf_file}_fold${fold}.log &
  done
  wait
fi

# classfication test
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2"
  python local/main.py --config conf/vhubert_lstm_drop0.2.yaml --gpu 0 --test_flag >> log/vhubert_lstm_drop0.2.log
fi

echo 'Done'
