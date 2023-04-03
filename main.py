import os
import time
import datetime
import pickle
import threading
import logging
import argparse

import math
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.server import Server

from robustbench.model_zoo.enums import ThreatModel

corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow',
    'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


# corruptions=['gaussian_noise']


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # dataset loading, output dir
    parser.add_argument('--data_dir', default='/data2/yongcan.yu/datasets',
                        help='the path of data to download and load')
    parser.add_argument('--imagenetc_dir', default='/data2/yongcan.yu/datasets/ImageNet-C',
                        help='the dir of imagenetc dataset')
    # parser.add_argument('--imagenetc_mode', default='full', choices=['full', 'part'])
    parser.add_argument('--ckpt_dir', default='./ckpt', help='the path of model to download and load')
    parser.add_argument('--output', default='./output/', help='the output directory of this experiment')
    parser.add_argument('--threat_model',
                        type=str,
                        default='Linf',
                        choices=[x.value for x in ThreatModel])
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'])
    # method
    parser.add_argument('--algorithm', default='eata', type=str,
                        choices=['source', 'norm', 'eata', 'tent', 'cotta', 'tent_ps', 'tent_psp', 'eata_m', 'ema',
                                 'T3A'],
                        help='eata or eta or tent')

    # general parameters, dataloader parameters
    parser.add_argument('--log_name', default='output.txt', type=str)
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default='7', type=str, help='GPU id to use.')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')

    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # model parameters
    parser.add_argument('--arch', default='Standard_R50',
                        choices=['Standard_R50', 'vit_base_patch16_224', 'visformer_small'], type=str,
                        help='the default model architecture')

    # dataset settings
    parser.add_argument('--severity', default=5, type=int, help='corruption level of test(val) set.')

    # eata settings
    parser.add_argument('--fisher_clip_by_norm', type=float, default=10.0, help='Clip fisher before it is too large')
    parser.add_argument('--fisher_size', default=2000, type=int,
                        help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000.,
                        help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000) * 0.40,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05,
                        help='epsilon in Eqn. (5) for filtering redundant samples')

    # 'continual' means the model parameters will never be reset, also called online adaptation; 'each_shift_reset'
    # means after each type of distribution shift, e.g., ImageNet-C Gaussian Noise Level 5, the model parameters will
    # be reset.

    # optimize hyperparameters
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD'],
                        help='the optimizer used under adaptation')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    # CoTTA parameters
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--RST', type=float, default=0.01)
    parser.add_argument('--ap', type=float, default=0.92)

    # FL parameters
    parser.add_argument('--local_batches', default=50, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--Federated', default=True, type=bool, help='Federated test time adaptation or not')
    # parser.add_argument('--dataloder_path', default='./iter_dataloders', type=str, help='the path to save and load fixed dataloders')
    parser.add_argument('--Fed_algorithm', default='FedAvg', type=str, choices=['FedAvg', 'FedProx', 'FedBNM'],
                        help='the algorithm used for Federated Learning')

    # server training parameters
    parser.add_argument('--train_server', default=True, type=bool, help='train the server model or not')
    parser.add_argument('--server_epochs', default=1, type=int, help='number of total epochs to run on server')
    parser.add_argument('--server_batch_size', default=64, type=int, help='server batch size')
    parser.add_argument('--server_lr', default=5e-4, type=float, help='server learning rate')
    parser.add_argument('--server_update_mode', default='EBN', type=str, choices=['BN', 'EBN', 'all', 'linear'],
                        help='server update mode')

    # FedProx parameters
    parser.add_argument('--mu', default=0.01, type=float,
                        help='the weight of loss of regularization')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.common_corruptions = corruptions
    # modify log_path to contain current time
    args.log_path = os.path.join(args.output, args.dataset, args.Fed_algorithm + '_' + args.algorithm,
                                 str(args.Federated) + '_' + str(args.local_batches))
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.log_path,
                              str(datetime.datetime.now().strftime(
                                  "%Y-%m-%d_%H:%M:%S")) + args.server_update_mode + '_' + args.log_name),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p",
        force=True)

    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message);
    logging.info(message)

    print(args)
    logger.info(args)

    # for config in configs:
    #     print(config); logging.info(config)
    print()

    # initialize federated learning 
    central_server = Server(args)
    central_server.setup()

    # do federated learning
    central_server.fit()

    # save resulting losses and metrics
    # with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
    #     pickle.dump(central_server.results, f)

    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message)
    logging.info(message)
    exit()
