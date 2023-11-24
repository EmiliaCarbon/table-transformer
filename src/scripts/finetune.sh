#!/bin/bash

ulimit -n 65536
ulimit -a

#python -m torch.distributed.launch --nproc_per_node 8 main.py \
#--data_type 'structure' \
#--config_file './structure_config_finetune.json' \
#--data_root_dir '/home/suqi/dataset/Pub_Syn_Union_1019/' \
#--batch_size 4 \
#--model_load_path '/home/suqi/model/TATR/TATR-v1.1-All-msft.pth' \
#--load_weights_only \
#--model_save_dir '/home/suqi/model/TATR/finetune'

# 从断点开始训练

#while true
#do
#  python -m torch.distributed.launch --nproc_per_node 8 main.py \
#  --data_type 'structure' \
#  --config_file './structure_config_finetune.json' \
#  --data_root_dir '/home/suqi/dataset/Pub_Fin_Syn_Union_Clean/' \
#  --batch_size 4 \
#  --model_load_path '/home/suqi/model/TATR/finetune/20231025080002/model.pth' \
#  --model_save_dir '/home/suqi/model/TATR/finetune/20231025080002' \
#  --without_dir_suffix \
#  --epochs 20
#
#  if [ $? == 0 ]; then
#    break
#  fi
#  sleep 100
#done
#
#sleep 100
#killall python
#sleep 100

while true
do
  python -m torch.distributed.launch --nproc_per_node 8 main.py \
  --data_type 'structure' \
  --config_file './structure_config_finetune_overlap.json' \
  --data_root_dir '/home/suqi/dataset/Pub_Fin_Syn_Union_Clean/' \
  --batch_size 4 \
  --model_load_path '/home/suqi/model/TATR/finetune/20231027181701/model.pth' \
  --model_save_dir '/home/suqi/model/TATR/finetune/20231027181701' \
  --without_dir_suffix \
  --epochs 30

  if [ $? == 0 ]; then
    break
  fi
  sleep 100
done

echo "### train finish ###"
