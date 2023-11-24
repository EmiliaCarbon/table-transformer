#!/bin/bash

cd ../

original_model="/home/suqi/model/TATR/TATR-v1.1-All-msft.pth"
finetune_v1_model="/home/suqi/model/TATR/finetune/20231016093942/model_best.pth"
finetune_v2_model="/home/suqi/model/TATR/finetune/20231020024934/model_best.pth"
finetune_v3_model="/home/suqi/model/TATR/finetune/20231025080002/model_best.pth"
finetune_v4_model="/home/suqi/model/TATR/finetune/20231027181701/model_best.pth"
finetune_v5_model="/home/suqi/model/TATR/finetune/train_finetune_resnet34/model_best.pth"

python main.py \
  --mode eval \
  --data_type structure \
  --config_file "structure_config.json" \
  --backbone "resnet34" \
  --data_root_dir "/home/suqi/dataset/temp/Pub_Fin_Syn_split/pubset" \
  --model_load_path "/home/suqi/model/TATR/finetune/train_finetune_resnet34/model_best.pth" \
  --table_words_dir "" \
  --num_workers 4 \
  --batch_size 16 \
  --device 2 \
#  --debug \
#  --debug_save_dir "/home/suqi/dataset/temp/visualize_v5/synset"
