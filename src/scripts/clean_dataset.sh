#!/bin/bash

ulimit -n 65536
ulimit -a

python clean_dataset.py \
--config_file './structure_config_finetune.json' \
--data_root_dir '/home/suqi/dataset/Pub_Fin_Syn_Union' \
--batch_size 1 \
--model_load_path '/home/suqi/model/TATR/finetune/20231020024934/model_best.pth' \
--load_weights_only
