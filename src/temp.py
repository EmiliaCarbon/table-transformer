import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm

data_root = '/home/suqi/dataset/pubtables-1m/PubTables-1M-Structure'
target_root = '/home/suqi/dataset/pubtables-1m/PubTables-1M-Structure-Clean'
os.makedirs(target_root, exist_ok=True)

with open(os.path.join('./statistics/filtered_images.txt'), 'r') as f:
    del_images_union = set([line.strip() for line in f.readlines()])

for dir_name in ['images', 'train', 'val', 'test']:
    dir_path = os.path.join(data_root, dir_name)
    os.makedirs(os.path.join(target_root, dir_name), exist_ok=True)
    del_count = 0
    for name in tqdm(os.listdir(dir_path)):
        if '_COL_' in name:
            base_name = name.split('_COL_')[0]
        elif '_CELL_' in name:
            base_name = name.split('_CELL_')[0]
        else:
            base_name = name.split('.')[0]

        if (base_name + '.jpg') in del_images_union:
            del_count += 1
        else:
            os.link(os.path.join(dir_path, name), os.path.join(target_root, dir_name, name))

    print(del_count)

for dir_name in ['images', 'train', 'val', 'test']:
    dir_path = os.path.join(target_root, dir_name)
    with open(os.path.join(target_root, f'{dir_name}_filelist.txt'), 'w') as f:
        for name in sorted(os.listdir(dir_path)):
            f.write(f'{dir_name}/{name}\n')
