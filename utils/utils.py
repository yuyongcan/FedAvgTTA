import os
import sys
import logging
import random

import numpy as np
import torch

import torch.nn as nn

import math
from methods.tent import softmax_entropy
from utils.cli_utils import accuracy as batch_acc,ProgressMeter
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def mean(items):
    return sum(items)/len(items)


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger
    

def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_flops(module: nn.Module, size, skip_pattern, device):
    # print(module._auxiliary)
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("init hool for", name)
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size).to(device))
        module.train(mode=training)
        # print(f"training={training}")
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops

def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param

def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None,top1=None,top5=None,ent=None,logger=None):
    from collections import defaultdict
    num_predict=defaultdict(int)
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    progress = ProgressMeter(
        n_batches,
        [top1, top5, ent],
        prefix='Test: ', logger=logger)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].cuda()
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].cuda()

            output = model(x_curr)
            # acc += (output.max(1)[1] == y_curr).float().sum()

            acc, acc5 = batch_acc(output, y_curr, topk=(1, 5))
            entropy = softmax_entropy(output).mean(0)
            top1.update(acc[0], x_curr.size(0))
            top5.update(acc5[0], x_curr.size(0))
            ent.update(entropy, x_curr.size(0))

            if counter %5==0:
                progress.display(counter)

            class_stat=output.max(1)[1].unique(return_counts=True)
            for i in range(class_stat[1].size()[0]):
                num_predict[class_stat[0][i].item()]+=class_stat[1][i].item()

    print(sorted(num_predict.items(), key=lambda e: e[1],reverse=True))
    return top1.avg/100