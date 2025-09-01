#!/bin/bash
. ./path.sh || exit 1;

# stage
if [ $# -ne 1 ]; then
  echo "Usage: $0 <stage>"
  exit 1
fi
stage=$1

# classfication
if [ ${stage} -eq 1 ]; then
  echo "stage 1"
  conf_files=(sharecrossAV_resnetseq10_tsnsegsplit_cosineatt)

  len=${#conf_files[@]}
  for ((i=0; i<${len}; i++)); do
    conf_file=${conf_files[i]}
    gpu_id=$((i % 4 + 1))
    echo "python local/main.py --config conf/${conf_file}.yaml --gpu ${gpu_id} --fold 0 >> log/${conf_file}.log"
    python local/main.py --config conf/${conf_file}.yaml --gpu ${gpu_id} --fold 0 >> log/${conf_file}.log &
  done
  wait
  echo "Done"
fi


# classfication test
if [ ${stage} -eq 2 ]; then
  echo "stage 2"
  con_files=(sharecrossAV_resnetseq10_tsnsegsplit_cosineatt)
  len=${#con_files[@]}
  for ((i=0; i<${len}; i++)); do
    conf_file=${con_files[i]}
    gpu_id=$((i % 4))
    echo "python local/main.py --config conf/${conf_file}.yaml --gpu ${gpu_id}  --test_flag >> log/${conf_file}.log &"
    python local/main.py --config conf/${conf_file}.yaml --gpu ${gpu_id} --fold 0 --test_flag >> log/${conf_file}.log &
  done
  wait
fi


echo 'Done'
