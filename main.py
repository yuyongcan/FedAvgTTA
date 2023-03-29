import os
import time
import datetime
import pickle
import threading
import logging
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.server import Server

from robustbench.model_zoo.enums import ThreatModel

corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow',
     'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # dataset loading, output dir
    parser.add_argument('--data_dir', default='/data/yongcan.yu/datasets', help='the path of data to download and load')
    parser.add_argument('--imagenetc_dir', default='/data/yongcan.yu/datasets/ImageNet-C',
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
    parser.add_argument('--algorithm', default='cotta', type=str,
                        choices=['source', 'norm', 'eata', 'tent', 'cotta', 'tent_ps','tent_psp','eata_m','ema'],
                        help='eata or eta or tent')
    # parser.add_argument('--parameters_update', default='grad', type=str,
    #                     choices=['BN', 'CLS', 'bias', 'all', 'random','grad'],
    #                     help='the parameters will be updated')
    # parser.add_argument('--random_update', default=0., type=float,
    #                     # choices=['BN', 'CLS', 'bias', 'all', 'random'],
    #                     help='the proportion of the parameters will be updated')
    # parser.add_argument('--grad_threshold', type=float, default=5e-5)
    # general parameters, dataloader parameters
    parser.add_argument('--log_name',default='FL.out',type=str)
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default='7', type=str, help='GPU id to use.')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # model parameters
    parser.add_argument('--arch', default='vit_base_patch16_224', choices=['Standard_R50','vit_base_patch16_224','visformer_small'],type=str, help='the default model architecture')

    # experiment mode setting
    # parser.add_argument('--exp_type', default='full', type=str,
    #                     choices=["continual", "A_S", "batch_level_corruption", 'full', 'adapt_ten_times'],
    #                     help='continual or each_shift_reset')

    # dataset settings
    # parser.add_argument('--corruption_order_index', type=int, default=0, help='the order of the corruptions to adapt')
    parser.add_argument('--severity', default=5, type=int, help='corruption level of test(val) set.')
    # parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    # parser.add_argument('--rotation', default=False, type=bool,
    #                     help='if use the rotation ssl task for training (this is TTTs dataloader).')

    # # eata settings
    # parser.add_argument('--fisher_clip_by_norm', type=float, default=10.0, help='Clip fisher before it is too large')
    # parser.add_argument('--fisher_size', default=2000, type=int,
    #                     help='number of samples to compute fisher information matrix.')
    # parser.add_argument('--fisher_alpha', type=float, default=2000.,
    #                     help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    # parser.add_argument('--e_margin', type=float, default=math.log(1000) * 0.40,
    #                     help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    # parser.add_argument('--d_margin', type=float, default=0.05,
    #                     help='\epsilon in Eqn. (5) for filtering redundant samples')

    # 'cotinual' means the model parameters will never be reset, also called online adaptation;
    # 'each_shift_reset' means after each type of distribution shift, e.g., ImageNet-C Gaussian Noise Level 5, the model parameters will be reset.

    # optimize hyper parameters
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

    #FL parameters
    parser.add_argument('--local_batches', default=20, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--Federated', default=True, type=bool, help='Federated test time adaptation or not')
    return parser.parse_args()

if __name__ == "__main__":


    args=get_args()
    args.common_corruptions=corruptions
    # modify log_path to contain current time
    args.log_path = os.path.join(args.output,args.algorithm,str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=args.log_path, filename_suffix="FL")
    # tb_thread = threading.Thread(
    #     target=launch_tensor_board,
    #     args=([log_config["log_path"], log_config["tb_port"], log_config["tb_host"]])
    #     ).start()
    # time.sleep(3.0)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.log_path, args.log_name),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")
    
    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)

    logger.info(args)

    # for config in configs:
    #     print(config); logging.info(config)
    print()

    # initialize federated learning 
    central_server = Server(writer, args)
    central_server.setup()

    # do federated learning
    central_server.fit()

    # save resulting losses and metrics
    # with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
    #     pickle.dump(central_server.results, f)
    
    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); exit()

