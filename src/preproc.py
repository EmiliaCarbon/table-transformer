from collections import defaultdict
from pathlib import Path
import os
from tqdm import tqdm
import pdb
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
from mpi4py import MPI


def denoise_molecular(image, dilate_degree=15):
    """
    @param image: np.ndarray, [h, w, ...]
    @param dilate_degree:
    remove the noise point in an image
    """
    fill_value = image.max()
    assert image.dtype == np.uint8 or fill_value > 2

    # binarize
    threshold = 250
    bin_image = np.mean(image, axis=-1).astype(np.uint8)
    mask = bin_image > threshold
    bin_image[mask] = 0
    bin_image[~mask] = 1

    kernel = np.ones(shape=(2, 2), dtype=np.int8)
    dilated_bin_image = cv2.dilate(bin_image, kernel, dilate_degree, iterations=dilate_degree).astype(np.uint8)
    # calculate connected domain, remove too small domain
    size_threshold = 0.5

    label, n_dom = ndimage.label(dilated_bin_image)
    dom_size = np.bincount(label.reshape(-1))
    max_dom = np.max(dom_size[1:])
    erase_dom = np.arange(0, n_dom + 1)[dom_size < int(max_dom * size_threshold)]
    dilated_bin_image[np.isin(label, erase_dom)] = 0

    # mask and cut
    x_range, y_range = np.where(dilated_bin_image == 0)
    image[x_range, y_range, :] = fill_value
    bin_image[x_range, y_range] = 0
    del x_range, y_range
    x_range, y_range = np.where(bin_image == 1)
    x_min, y_min = map(lambda x: x.min(), (x_range, y_range))
    x_max, y_max = map(lambda x: x.max(), (x_range, y_range))

    return image[x_min: x_max + 1, y_min: y_max + 1]


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    mole_root = '/home/suqi/dataset/MolScribe/preprocessed'
    mole_files = list(sorted(os.listdir(mole_root)))
    tasks_per_cpu = np.array([len(mole_files) // size] * size)
    tasks_per_cpu[: len(mole_files) - np.sum(tasks_per_cpu)] += 1
    indices = [0] + np.cumsum(tasks_per_cpu).tolist()
    print(indices)
    mole_files = mole_files[indices[rank]: indices[rank + 1]]

    new_mole_root = '/home/suqi/dataset/MolScribe/preprocessed'
    if rank == 0:
        os.makedirs(new_mole_root, exist_ok=True)

    for name in tqdm(mole_files):
        if os.path.exists(os.path.join(new_mole_root, name)):
            continue
        mole_image = cv2.cvtColor(cv2.imread(os.path.join(mole_root, name)), cv2.COLOR_RGB2BGR)
        mole_image = denoise_molecular(mole_image, 15)
        cv2.imwrite(os.path.join(new_mole_root, name), mole_image)

# def cut_white_border(image: np.ndarray, threshold=250) -> np.ndarray:
#     """
#     image: binary image, (H x W)
#     """
#     x_range, y_range = np.where(image < threshold)
#     x_min, y_min = map(lambda x: np.min(x), (x_range, y_range))
#     x_max, y_max = map(lambda x: np.max(x), (x_range, y_range))
#     return image[x_min: x_max + 1, y_min: y_max + 1]
#
#
# data_root = '/home/suqi/dataset/MolScribe/'
# target_folder = 'preprocessed'
# os.makedirs(os.path.join(data_root, target_folder), exist_ok=True)
#
# patterns = [
#     # '.png', '.jpg',
#     '.TIF',
#     # '.tiff', '.bmp'
# ]
#
# # png_folders = [
# #     'indigo_validation_set_examples/images',
# #     'perturb/CLEF_pertubations/*',
# #     'perturb/STAKER',
# #     'perturb/UOB_pertubations/*',
# #     'perturb/USPTO_pertubations',
# #     'synthetic/chemdraw',
# #     'synthetic/indigo',
# #     'uspto_validation_set_examples/images',
# #     'valko_testset_results/image_results'
# # ]
#
# dataset_counter = defaultdict(lambda: 0)
# data_dict = {}
# for pattern in patterns:
#     image_list = sorted(Path(data_root).rglob('*' + pattern))
#     print(len(image_list))
#     for image_path in tqdm(image_list, desc=f'Now: {pattern}'):
#         try:
#             dataset = str(image_path).strip().split('/')[5]
#             data_idx = dataset_counter[dataset]
#             dataset_counter[dataset] = data_idx + 1
#
#             save_name = dataset + '_' + str(data_idx).rjust(7, '0') + '.png'
#             data_dict[save_name] = image_path
#             if os.path.exists(os.path.join(data_root, target_folder, save_name)):
#                 continue
#
#             image = np.array(Image.open(os.path.join(data_root, image_path)))
#             if len(image.shape) == 3:
#                 image = image.mean(axis=-1)
#
#             if pattern == '.tiff' or pattern == '.bmp':
#                 width = image.shape[1]
#                 image = image[:, width * 3 // 4:]
#
#             if image.max() <= 1.1:
#                 threshold = 250 / 255
#                 image = cut_white_border(image.astype(np.float16), threshold)
#                 image = (image * 255).astype(np.uint8)
#             else:
#                 threshold = 250
#                 image = cut_white_border(image.astype(np.uint8), threshold)
#
#             image = np.tile(image[..., None], 3)
#             Image.fromarray(image).save(os.path.join(data_root, target_folder, save_name))
#         except Exception as e:
#             print(f'Exception at: {image_path}')
#             pdb.set_trace()
