import os
import sys
import random
from collections import defaultdict
import torch
import numpy as np

sys.path.append("../detr")
from docai_util import image_replace, box_shrink
from tqdm import tqdm
import bisect
from PIL import Image
from mpi4py import MPI


def rearrange_cell(cell):
    """
    将拉直的cell重新按列优先排列，如果失败则返回None
    @param cell: [K, 4]
    @return [N, M, 4]
    """
    if len(cell) == 0:
        return None
    x0, x1 = cell[:, 0], cell[:, 2]
    diff_x0 = ((x0 - x0[0]) < 1).astype(np.uint8)
    diff_x1 = ((x1 - x1[0]) < 1).astype(np.uint8)
    n_row_0 = np.count_nonzero(diff_x0)
    n_row_1 = np.count_nonzero(diff_x1)
    if n_row_0 != n_row_1 or len(cell) % n_row_0 != 0:
        return None
    n_row, n_col = n_row_0, len(cell) // n_row_0
    cell = cell.reshape((n_col, n_row, 4)).transpose(1, 0, 2)
    return cell


def load_cells(cell_top_k, col_top_k, dataset='pubtable'):
    # load cell map
    cell_map = torch.load(f'./statistics/cell_shapes_{dataset}.pth')

    # 在单个表格上的统计量
    xml_paths, cells, matrix_cells = [], [], []
    for idx, (path, cell) in tqdm(enumerate(cell_map.items()), desc='Loading cell maps: '):
        xml_paths.append(path)
        cells.append(cell)
        matrix_cells.append(rearrange_cell(cell))

    # 在每个cell上的统计量
    cells_merge, cells_xml_indices = [], []
    cells_concat = []
    for idx, (path, cell) in tqdm(enumerate(cell_map.items()), desc='Calculating cell statistic: '):
        cells_concat.append(cell)
        cells_merge.extend(range(len(cell)))
        cells_xml_indices.extend([idx] * len(cell))

    cells_merge, cells_xml_indices = map(lambda x: np.array(x), (cells_merge, cells_xml_indices))
    cells_concat = np.concatenate(cells_concat, axis=0)

    # 计算每个单元格的宽高
    widths = cells_concat[:, 2] - cells_concat[:, 0]
    heights = cells_concat[:, 3] - cells_concat[:, 1]
    del cells_concat

    # 用高度和面积综合排序
    indices_height = np.argsort(heights, axis=0)
    indices_area = np.argsort(heights * widths, axis=0)

    # 计算每个单元格的分数
    cells_score = np.zeros_like(indices_height)
    cells_score[indices_height] += np.arange(len(indices_height))
    cells_score[indices_area] += np.arange(len(indices_area))

    # 获取单张表格图片中每列的分数
    start = 0
    cols_merge, cols_xml_indices, cols_score = [], [], []
    for idx, (path, cell) in tqdm(enumerate(cell_map.items()), desc='Calculating col statistic: '):
        if matrix_cells[idx] is not None:
            n_row, n_col = matrix_cells[idx].shape[:2]
            matrix_score = cells_score[start: start + len(cell)].reshape((n_col, n_row)).transpose(1, 0)
            if n_row >= 4:
                col_score = matrix_score.min(axis=0)
            else:
                col_score = np.zeros(shape=(n_col,))
            cols_merge.extend(range(n_col))
            cols_xml_indices.extend([idx] * n_col)
            cols_score.append(col_score)
        start += len(cell)
    cols_merge, cols_xml_indices = map(lambda x: np.array(x), (cols_merge, cols_xml_indices))
    cols_score = np.concatenate(cols_score, axis=0)

    # 选择分数最高的若干个cell
    sorted_indices = cells_score.argsort()[::-1][:cell_top_k]
    cells_merge, cells_xml_indices, cells_score = map(lambda x: x[sorted_indices],
                                                      (cells_merge, cells_xml_indices, cells_score))

    # 选择分数最高的若干个列
    col_sorted_indices = cols_score.argsort()[::-1][:col_top_k]
    cols_merge, cols_xml_indices, cols_score = map(lambda x: x[col_sorted_indices],
                                                   (cols_merge, cols_xml_indices, cols_score))

    # 整理并保存到字典中
    xml_remain = defaultdict(lambda: {})

    # 被选中的xml索引
    for xml_idx in np.unique(cells_xml_indices):
        mask = cells_xml_indices == xml_idx
        xml_remain[xml_paths[xml_idx]].update({
            'cell': {
                'indices': cells_merge[mask],
                'scores': cells_score[mask],
                'data': cells[xml_idx]
            }
        })

    for xml_idx in np.unique(cols_xml_indices):
        mask = cols_xml_indices == xml_idx
        xml_remain[xml_paths[xml_idx]].update({
            'col': {
                'indices': cols_merge[mask],
                'scores': cols_score[mask],
                'data': matrix_cells[xml_idx]
            }
        })

    return xml_remain


def load_molecular_images():
    image_names = []
    image_shapes = []

    for key, val in torch.load('./statistics/molecular_image_shape.pth').items():
        image_names.append(key)
        image_shapes.append(val)

    image_names = np.array(image_names)
    image_shapes = np.stack(image_shapes, axis=0)

    whr_ratios = image_shapes[:, 0] / image_shapes[:, 1]
    whr_indices = np.argsort(whr_ratios)
    return image_names[whr_indices], image_shapes[whr_indices], whr_ratios[whr_indices]


def load_optimal_mole(bbox,
                      mole_ratios,
                      mole_shapes,
                      search_radius=20):
    box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    whr = box_width / box_height
    # 找出与当前cell长宽比最接近的分子图片
    mole_idx = bisect.bisect(mole_ratios, whr)
    cand_indices = np.arange(max(mole_idx - search_radius, 0),
                             min(mole_idx + search_radius, len(mole_ratios)))

    # 计算这些候选分子图填到cell里需要进行的缩放的大小
    scales = (np.array([box_width, box_height]) / mole_shapes[cand_indices]).min(axis=-1)
    scales[scales < 1] = 1 / scales[scales < 1]

    # 缩放越接近1的分子有更大概率被抽到
    prob_no_norm = np.exp(-scales)
    prob = prob_no_norm / np.sum(prob_no_norm)
    idx = np.random.choice(cand_indices, size=1, replace=False, p=prob)[0]
    return idx


def generate():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    table_image_root = '/home/suqi/dataset/FinTabNet.c/FinTabNet.c-Structure/images'
    mole_image_root = '/home/suqi/dataset/MolScribe/preprocessed'
    save_dir = '/home/suqi/dataset/synthesis_fintable'

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    # xml_remain = load_cells(cell_top_k=1000000, col_top_k=5000)
    xml_remain = load_cells(cell_top_k=50000, col_top_k=300, dataset='fintable')
    mole_names, mole_shapes, mole_ratios = load_molecular_images()

    # 在长宽比最匹配的分子图像附近随机选择，该参数决定了选择的范围
    search_radius = 20

    # 多CPU运算
    xml_remain = list(sorted(xml_remain.items(), key=lambda x: x[0]))
    tasks_per_cpu = np.array([len(xml_remain) // size] * size)
    tasks_per_cpu[: len(xml_remain) - np.sum(tasks_per_cpu)] += 1
    indices = [0] + np.cumsum(tasks_per_cpu).tolist()
    xml_remain = xml_remain[indices[rank]: indices[rank + 1]]

    for xml_path, xml_info in tqdm(xml_remain, desc='Generating: '):
        try:
            table_base_name = xml_path.split('/')[-1].split('.xml')[0]
            table_image_source = np.array(Image.open(os.path.join(table_image_root, table_base_name + '.jpg')))

            if 'cell' in xml_info.keys():
                data = xml_info['cell']['data']
                indices = xml_info['cell']['indices']
                scores = xml_info['cell']['scores']

                # 重复若干次以生成更多数据
                repeat = np.clip(len(indices) // 3 + 1, 1, 10)
                for rep in range(repeat):
                    table_image = table_image_source.copy()
                    # dropout some cells if necessary
                    max_remain = min(np.random.randint(len(data) // 5, len(data) // 3 + 1) + 1, len(indices))

                    # 将各个cell的分数作为概率，分数更高的cell更可能被留下
                    remain_indices = np.random.choice(indices,
                                                      max_remain,
                                                      replace=False,
                                                      p=scores / np.sum(scores))

                    for bbox in data[remain_indices]:
                        # 收缩cell以防止其中包含有网格线
                        bbox = box_shrink(table_image_source, bbox, 0.9)
                        idx = load_optimal_mole(bbox, mole_ratios, mole_shapes, search_radius)
                        mole_image = np.array(
                            Image.open(os.path.join(mole_image_root, str(mole_names[idx]))).convert('RGB'))
                        table_image = image_replace(table_image, bbox, mole_image, scale_shrink=0.8)

                    Image.fromarray(table_image).save(os.path.join(save_dir,
                                                                   f"{table_base_name}_CELL_{str(rep).rjust(2, '0')}.jpg"))
            if 'col' in xml_info.keys():
                data = xml_info['col']['data']
                indices = xml_info['col']['indices']
                scores = xml_info['col']['scores']
                n_row, n_col = data.shape[:2]

                # 重复若干次以生成更多数据
                repeat = np.clip((len(indices) * n_row) // 3 + 1, 1, 10)
                for rep in range(repeat):
                    table_image = table_image_source.copy()
                    # dropout some cols if necessary
                    max_remain = min(np.random.randint(max(n_col // 5, 1), min(4, n_col // 2) + 1), len(indices))

                    # 将各个列的分数作为概率，分数更高的列更可能被留下
                    remain_indices = np.random.choice(indices,
                                                      max_remain,
                                                      replace=False,
                                                      p=scores / np.sum(scores))

                    for bbox in data[:, remain_indices].reshape(-1, 4):
                        # 收缩cell以防止其中包含有网格线
                        bbox = box_shrink(table_image_source, bbox, 0.9)
                        idx = load_optimal_mole(bbox, mole_ratios, mole_shapes, search_radius)
                        mole_image = np.array(
                            Image.open(os.path.join(mole_image_root, str(mole_names[idx]))).convert('RGB'))
                        # 插入的分子图像在列上尽可能紧贴，因此设置scale_shrink为1
                        table_image = image_replace(table_image, bbox, mole_image, scale_shrink=1)

                    Image.fromarray(table_image).save(os.path.join(save_dir,
                                                                   f"{table_base_name}_COL_{str(rep).rjust(2, '0')}.jpg"))
        except Exception as e:
            pass


if __name__ == '__main__':
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    generate()
