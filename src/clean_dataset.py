"""
Copyright (C) 2021 Microsoft Corporation
"""
import builtins
import os
import argparse
import json
from datetime import datetime
import sys
import random
import numpy as np
from torch.utils.data import DataLoader

sys.path.append("../detr")
from models import build_model
import util.misc as utils

import table_datasets as TD
from table_datasets import PDFTablesDataset
import torch
import warnings

warnings.filterwarnings('ignore')


# def print(*args, **kwargs):
#     if dist.get_rank() == 0:
#         builtins.print(*args, **kwargs)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_dir',
                        required=True,
                        help="Root data directory for images and labels")
    parser.add_argument('--config_file',
                        required=True,
                        help="Filepath to the config containing the args")
    parser.add_argument('--backbone',
                        default='resnet18',
                        help="Backbone for the model")
    parser.add_argument(
        '--data_type',
        choices=['detection', 'structure'],
        default='structure',
        help="toggle between structure recognition and table detection")
    parser.add_argument('--model_load_path', help="The path to trained model")
    parser.add_argument('--load_weights_only', action='store_true')

    parser.add_argument('--device')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--val_max_size', type=int)
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


def get_transform(data_type, image_set):
    if data_type == 'structure':
        return TD.get_structure_transform(image_set)
    else:
        return TD.get_detection_transform(image_set)


def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
    else:
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map


def get_data(args):
    """
    Based on the args, retrieves the necessary data to perform training,
    evaluation or GriTS metric evaluation
    """
    # Datasets
    print("loading data")
    class_map = get_class_map(args.data_type)

    dataset_val = PDFTablesDataset(os.path.join(args.data_root_dir, "all"),
                                   get_transform(args.data_type, "val"),
                                   do_crop=False,
                                   max_size=args.val_max_size,
                                   include_eval=True,
                                   make_coco=True,
                                   image_extension=".jpg",
                                   xml_fileset="all_filelist.txt",
                                   class_map=class_map)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)
    return data_loader_val, dataset_val


def get_model(args, device):
    """
    Loads DETR model on to the device specified.
    If a load path is specified, the state dict is updated accordingly.
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.model_load_path:
        print("loading model from checkpoint")
        loaded_state_dict = torch.load(args.model_load_path,
                                       map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=True)
    return model, criterion, postprocessors


def evaluate_loss(args, model, criterion, device):
    """
    eval dataset loss
    """
    print("loading data")
    dataloading_time = datetime.now()
    data_loader, dataset = get_data(args)
    print("finished loading data in :", datetime.now() - dataloading_time)

    if args.model_load_path:
        checkpoint = torch.load(args.model_load_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

    print(f"Start Evaluation: {len(data_loader)} batches.")

    losses = {}

    with torch.no_grad():
        model.eval()
        criterion.eval()

        metric_logger = utils.MetricLogger(delimiter="  ")

        for samples, targets in metric_logger.log_every(data_loader, 1000, f'Test: '):
            samples = samples.to(device)
            img_paths = [t['img_path'] for t in targets]
            targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            losses[img_paths[0]] = loss_dict

    torch.save(losses, './statistics/losses.pth')


def main():
    cmd_args = get_args().__dict__
    config_args = json.load(open(cmd_args['config_file'], 'rb'))
    for key, value in cmd_args.items():
        if not key in config_args or not value is None:
            config_args[key] = value
    # config_args.update(cmd_args)
    args = type('Args', (object,), config_args)

    torch.cuda.set_device(0)

    print(args.__dict__)
    print('-' * 100)

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("loading model")
    device = torch.device('cuda', 0)
    model, criterion, postprocessors = get_model(args, device)

    evaluate_loss(args, model, criterion, device)


if __name__ == "__main__":
    main()
