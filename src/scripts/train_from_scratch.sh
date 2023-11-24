#!/bin/bash

ulimit -n 65536

# --backbone --CUDA_VISIBLE_DEVICES --server_socket --nnodes --node_rank --master_addr

# backbone as the first param
backbone=$1
exp_name="train_finetune_${backbone}"
err_file="logs/error_${exp_name}.txt"
touch $err_file

if [[ ! $2 =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Error CUDA_VISIBLE_DEVICES Format: $2"
  exit 1
fi

if [[ ! $3 =~ [0-9]+ ]]; then
  echo "Error Server Socket Format: $3"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$2
server_socket=$3
nnodes=$4
node_rank=$5
master_addr=$6

echo "Use Backbone: ${backbone}"
echo "Visible CUDA Devices: ${CUDA_VISIBLE_DEVICES}"
echo "DDP Server Socket: ${server_socket}"

nproc_per_node=$(grep -oE "[0-9]+" <<< ${CUDA_VISIBLE_DEVICES} | wc -l)
echo "Device Number: ${nproc_per_node}"

while true
do
  echo $(TZ=UTC-8 date +%Y-%m-%d" "%H:%M:%S) >> ${err_file}
  torchrun --nnodes=${nnodes} \
           --node_rank=${node_rank} \
           --nproc_per_node=${nproc_per_node} \
           --master_port=${server_socket} \
           --master_addr=${master_addr} \
           main.py \
              --data_type 'structure' \
              --config_file './structure_config_finetune.json' \
              --data_root_dir '/home/suqi/dataset/Pub_Fin_Syn_Union_Clean' \
              --batch_size 6 \
              --model_save_dir "/home/suqi/model/TATR/finetune/${exp_name}" \
              --without_dir_suffix \
              --model_load_path "/home/suqi/model/TATR/finetune/${exp_name}/model.pth" \
              --epochs 20 \
              --backbone "${backbone}" \
              2>>${err_file}

  curr_time=$(TZ=UTC-8 date +%Y-%m-%d' '%H:%M:%S)
  if [ $? == 0 ]; then
    echo "#### Training finished at time ${curr_time} ####"
    break
  fi
  echo "#### Exception happened at time ${curr_time}, see log file in ${err_file} ####" | tee -a ${err_file}
  echo -e "\n\n\n" | tee -a ${err_file}
  sleep 100
done

# bash scripts/train_from_scratch.sh "resnet34" "0,1,2,3,4,5,6,7" "22335" "2" "0" "10.148.0.13"
# bash scripts/train_from_scratch.sh "resnet34" "0,1,2,3,4,5,6,7" "22335" "2" "1" "10.148.0.13"
