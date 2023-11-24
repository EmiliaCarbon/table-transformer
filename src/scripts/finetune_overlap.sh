#!/bin/bash

ulimit -n 65536
ulimit -a

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
