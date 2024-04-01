#!/bin/bash

. ./path.sh || exit 1;

stage=3

# test data_processor.py
if [ ${stage} -eq 0 ]; then
  echo "stage 0"
  python -m pdb networks/dataset/data_processor.py
fi

# test dataset.py
if [ ${stage} -eq 1 ]; then
  echo "stage 1"
  python -m pdb networks/dataset/dataset.py
fi

# test load_data.py
if [ ${stage} -eq 2 ]; then
  echo "stage 2"
  python -m pdb networks/dataset/load_data.py
fi


# test main.py
if [ ${stage} -eq 3 ]; then
  echo "stage 3"
  python -m pdb local/main.py --config conf/clmrv_lstm_drop0.2.yaml --gpu 0 --fold 0
fi

# test tsn.py
if [ ${stage} -eq 4 ]; then
  echo "stage 4"
  python -m pdb networks/model/tsn/tsn.py
fi