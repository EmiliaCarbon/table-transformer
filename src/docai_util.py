from typing import Union
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou
from scipy import ndimage
from sklearn.cluster import KMeans
import torch.distributed as dist
import builtins
import os


def print(*args, **kwargs):
    if not dist.is_initialized() or int(os.environ["LOCAL_RANK"]) == 0:
        builtins.print(*args, **kwargs)


def is_color_image(image, diff_threshold=2):
    if len(image.shape) == 2:
        return False
    channel_diff = np.abs(image.max(axis=-1) - image.min(axis=-1)).mean()
    return channel_diff > diff_threshold


def binarize(image: np.ndarray, return_colors=False):
    """
    使用聚类算法二值化图像，返回的图像是 HxW 的0，1二值图像，前景为1，背景为0，同时返回前景和背景的颜色（可选）
    @param image: [h, w, 3]
    @param return_colors: 是否返回前景背景颜色
    """
    # 对于彩色图像或背景为灰色的图像，使用聚类算法二值化图像
    if is_color_image(image) or image.max() < 250:
        h, w = image.shape[:2]
        image = image.reshape(-1, 3)
        model = KMeans(n_clusters=2, tol=0.1, n_init='auto')
        model.fit(image)
        labels = model.labels_

        fore_pixels = image[labels == 1]
        back_pixels = image[labels == 0]
        back_center = model.cluster_centers_[0]
        # 取和背景中心距离最小的若干个像素的平均值做为背景色
        back_chosen = np.argsort(((back_pixels - back_center) ** 2).sum(axis=-1))[len(back_pixels) // 5:
                                                                                  len(back_pixels) // 2]
        back_color = np.mean(back_pixels[back_chosen], axis=0).astype(np.uint8)
        # 取和背景中心距离最大的若干个像素的平均值做为前景色，天才
        if len(fore_pixels):
            fore_chosen = np.argsort(((fore_pixels - back_center) ** 2).sum(axis=-1))[::-1][
                          : max(len(fore_pixels) // 4, 10)]
            fore_color = np.mean(fore_pixels[fore_chosen], axis=0).astype(np.uint8)
        else:
            fore_color = np.array([0, 0, 0], dtype=np.uint8)

        # 较多的簇中心设置为背景
        bin_image = labels.reshape((h, w)).astype(np.uint8)
        unique, counts = np.unique(labels, return_counts=True)
        if unique[np.argmax(counts)] == 1:
            bin_image = 1 - bin_image
            fore_color, back_color = back_color, fore_color
    # 对于黑白图像，直接用阈值二值化
    else:
        bin_image = np.zeros(shape=image.shape[:2], dtype=np.uint8)
        mask = np.any(image < 250, axis=-1)
        bin_image[mask] = 1
        fore_color = np.array([0, 0, 0], dtype=np.uint8)
        back_color = np.array([255, 255, 255], dtype=np.uint8)

    if return_colors:
        return bin_image, fore_color, back_color
    return bin_image


def add_margin_to_image(pil_img, padding, color=(255, 255, 255)):
    """
    Image padding as part of TSR pre-processing to prevent missing table edges
    """
    if pil_img is None:
        return None
    if isinstance(padding, int):
        left = right = top = bottom = padding
    elif len(padding) == 2:
        left = right = padding[0]
        top = bottom = padding[1]
    elif len(padding) == 4:
        left, top, right, bottom = padding

    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def bboxes_to_cells(bboxes: np.ndarray,
                    labels: np.ndarray,
                    col_index=1,
                    row_index=2,
                    exclude_indices=(3, 4),
                    exclude_threshold=0.001):
    cols = bboxes[labels == col_index][:, None, :]  # (N, 1, 4)
    rows = bboxes[labels == row_index][None, :, :]  # (1, M, 4)
    n, m = cols.shape[0], rows.shape[1]
    x_min = cols[:, :, 0].repeat(m, axis=1)  # (N, 1)
    y_min = rows[:, :, 1].repeat(n, axis=0)  # (1, M)
    x_max = cols[:, :, 2].repeat(m, axis=1)  # (N, 1)
    y_max = rows[:, :, 3].repeat(n, axis=0)  # (1, M)
    cells = np.stack([x_min, y_min, x_max, y_max], axis=-1).reshape((n * m, -1))  # (N * M, 4)

    exclude_rows = bboxes[np.isin(labels, exclude_indices)]  # (K, 4)

    iou = box_iou(torch.from_numpy(cells), torch.from_numpy(exclude_rows)).numpy()
    cells = cells[np.all(iou < exclude_threshold, axis=1)]
    return cells


def image_replace(source: np.ndarray,
                  bbox: Union[np.ndarray, list],
                  target: np.ndarray,
                  scale_shrink=0.9):
    """
    传入此函数的bbox不会经过shrink，因此在传入之前最好首先调用'box_shrink'函数以避免把边界线覆盖的情况
    source: (h, w, ...)
    bbox: (x1, y1, x2, y2)
    target: (h1, w1, ...)
    """
    source, target = map(lambda x: np.array(x), (source, target))

    assert 0.5 <= scale_shrink <= 1
    source = source[..., :3]
    if target.shape[-1] > 3:
        target = target[..., :3]
    elif target.shape[-1] < 3:
        target = np.repeat(target[..., 0], 3, -1)

    bbox = list(map(int, bbox))
    h1, w1 = target.shape[:2]
    box_height, box_width = int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0])
    cell_source = source[bbox[1]: bbox[3], bbox[0]: bbox[2]]

    # 获取cell的二值化图，以及其中的前景、背景色
    cell_bin, cell_fore_color, cell_back_color = binarize(cell_source, return_colors=True)
    # 获取target的二值化图，并用cell的前景、背景颜色代替
    targ_bin = np.all(target < 250, axis=-1)
    target[targ_bin] = cell_fore_color
    target[~targ_bin] = cell_back_color

    scale_rate = min(box_width / w1, box_height / h1) * 0.9
    # random resize in [scale_shrink * scale_rate, scale_rate]
    scale = np.random.rand() * (1 - scale_shrink) * scale_rate + scale_shrink * scale_rate
    scaled_y, scaled_x = int(h1 * scale), int(w1 * scale)
    target = np.array(Image.fromarray(target).resize((scaled_x, scaled_y)))

    spacing_x, spacing_y = box_width - scaled_x, box_height - scaled_y
    # pad_left, pad_top = int(np.random.rand() * spacing_x), int(np.random.rand() * spacing_y)

    # 计算表格中的原数据中心位于cell的何处，并作为插入图像的中心位置
    fore_ys, fore_xs = np.where(cell_bin == 1)
    if len(fore_xs):
        fore_center_x = (fore_xs.max() + fore_xs.min()) // 2
        fore_center_y = (fore_ys.max() + fore_ys.min()) // 2
    else:
        fore_center_x = (bbox[2] - bbox[0]) // 2
        fore_center_y = (bbox[3] - bbox[1]) // 2
    # fore_center_x = int(fore_xs.mean())
    # fore_center_y = int(fore_ys.mean())

    pad_left = np.clip(fore_center_x - scaled_x // 2, spacing_x // 6, 5 * spacing_x // 6)
    pad_top = np.clip(fore_center_y - scaled_y // 2, spacing_y // 6, 5 * spacing_y // 6)

    # mask with background color and replace
    source[bbox[1]: bbox[3], bbox[0]: bbox[2]] = cell_back_color

    source[
    bbox[1] + pad_top: bbox[1] + pad_top + scaled_y,
    bbox[0] + pad_left: bbox[0] + pad_left + scaled_x
    ] = target
    return source


def _boundary_range(binary_line):
    """
    @param binary_line: 1-d binary line
    """
    start, end = 0, len(binary_line)
    label, n_dom = ndimage.label(binary_line)

    # split boundary
    split_const = 5
    head_bound = len(binary_line) / split_const
    tail_bound = (split_const - 1) * len(binary_line) / split_const

    if n_dom == 1:
        indices = np.where(label == 1)[0]
        mid = np.mean(indices)
        # this line is start line
        if mid < head_bound:
            start = indices[-1] + 1
        # this line is end line
        elif mid > tail_bound:
            end = indices[0]
    elif n_dom > 1:
        first_indices = np.where(label == 1)[0]
        last_indices = np.where(label == n_dom)[0]
        if np.mean(first_indices) < head_bound:
            start = first_indices[-1] + 1
        if np.mean(last_indices) > tail_bound:
            end = last_indices[0]
    return start, end


def box_shrink(image, bbox, line_rate_threshold=0.9):
    """
    收缩box以排除其中可能含有的边框线，image可以为彩色
    """
    assert image.dtype == np.uint8 or image.max() > 2

    x_min, y_min, x_max, y_max = list(map(int, bbox))
    box_image = image[y_min: y_max, x_min: x_max]
    bin_image = binarize(box_image)
    h, w = bin_image.shape

    # check line if exists
    row_sum = bin_image.sum(axis=1)
    col_sum = bin_image.sum(axis=0)

    row_lines = (row_sum >= int(w * line_rate_threshold)).astype(np.uint8)
    col_lines = (col_sum >= int(h * line_rate_threshold)).astype(np.uint8)

    col_range = _boundary_range(col_lines)
    row_range = _boundary_range(row_lines)

    x_min, y_min = x_min + col_range[0], y_min + row_range[0]
    x_max = x_min + col_range[1] - col_range[0]
    y_max = y_min + row_range[1] - row_range[0]

    return [x_min, y_min, x_max, y_max]
    # 进一步收缩几个像素以绝对避免框到表格线
    # return [x_min + 1, y_min + 1, x_max - 1, y_max - 1]


def image_stroke_size(image: np.ndarray, device='cpu', max_size=30) -> float:
    """
    计算一张图中线条的笔触大小
    @param image: np.ndarray, [h, w, ...]
    """
    fill_value = image.max()
    assert image.dtype == np.uint8 or fill_value > 2

    h, w = image.shape[:2]

    # binarize the box image
    bin_threshold = 250
    bin_image = np.mean(image, axis=-1).astype(np.uint8) if len(image.shape) == 3 else image.astype(np.uint8)
    mask = bin_image > bin_threshold
    bin_image[~mask] = 0
    bin_image[mask] = 1

    # (1, 1, h, w)
    kernel_sizes = list(range(1, max_size + 1))
    bin_image = torch.from_numpy(bin_image).to(device).to(torch.uint8)[None, None, ...]

    bin_pool = [torch.max_pool2d(bin_image, kernel_size=sz, stride=1, padding=sz // 2) for sz in kernel_sizes]
    bin_pool = [bin_pool[idx] if sz % 2 == 1 else bin_pool[idx][:, :, 1:, 1:] for idx, sz in kernel_sizes]
    bin_pool = torch.cat(bin_pool, dim=1)[0].detach().cpu().numpy().transpose(1, 2, 0)
    # (h, w, max_size)
    assert tuple(bin_pool.shape) == (h, w, len(kernel_sizes))

    # calculate the stroke size on every pixel point
    stroke_sizes = np.argmax(bin_pool, axis=-1)
    stroke_sizes[np.where(np.sum(bin_pool, axis=-1) == 0)] = max_size

    return np.mean(stroke_sizes[bin_image == 0])


if __name__ == '__main__':
    binary_line = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(_boundary_range(binary_line))
