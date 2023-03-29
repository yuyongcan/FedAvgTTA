import copy
import gc
import random

import math
from torch import optim
from tqdm.auto import tqdm
from collections import OrderedDict

# from .models import *
from .utils import *
from .client import Client

from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from methods import tent,cotta,eata
import timm

logger = logging.getLogger(__name__)

def setup_optimizer(params, args):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if args.optimizer == 'Adam':
        return optim.Adam(params,
                          lr=args.lr,
                          betas=(args.beta, 0.999),
                          weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        return optim.SGD(params,
                         lr=args.lr,
                         momentum=0.9,
                         dampening=0,
                         weight_decay=args.weight_decay,
                         nesterov=True)
    else:
        raise NotImplementedError

class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        adapt_model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """

    def __init__(self, writer, args):
        self.clients = None
        self._round = 0
        self.writer = writer
        self.device = "cuda:"+args.gpu
        if args.dataset == 'imagenet':
            # args.arch = 'Standard_R50'
            args.data_dir = '/data/yongcan.yu/datasets'
            self.num_rounds = 50000 // args.batch_size // args.local_batches + 1
        elif args.dataset in ['cifar10', 'cifar100']:
            args.data_dir = './data'
            if args.dataset == 'cifar10':
                args.arch = 'Standard'
                self.num_rounds = 10000 // args.batch_size // args.local_batches + 1
            elif args.dataset == 'cifar100':
                args.arch = 'Hendrycks2020AugMix_ResNeXt'
                self.num_rounds = 10000 // args.batch_size // args.local_batches + 1

        if args.arch in ['visformer_small', 'vit_base_patch16_224']:
            subnet = timm.create_model(args.arch, pretrained=True).to(self.device)
        else:
            subnet = load_model(args.arch, args.ckpt_dir,
                                args.dataset, ThreatModel.corruptions).to(self.device)


        if args.algorithm == 'tent':
            subnet = tent.configure_model(subnet)
            params, param_names = tent.collect_params(subnet)
            if args.dataset == 'imagenet':
                # optimizer = torch.optim.SGD(params, 0.00025 if args.arch=='Standard_R50' and args.exp_type=='full' else 0.001, momentum=0.9)
                optimizer = torch.optim.SGD(params,
                                            0.00025,
                                            momentum=0.9)
            else:
                optimizer = torch.optim.Adam(params, 0.001)
            adapt_model = tent.Tent(subnet, optimizer)
        elif args.algorithm == 'cotta':
            if args.dataset == 'imagenet':
                args.lr = 0.01
                args.step = 1
                args.beta = 0.9
                args.weight_decay = 0.
                args.momentum = 0.999
                args.rst = 0.01
                args.ap = 0.92
                args.optimizer = 'SGD'
            elif args.dataset == 'cifar10':
                args.lr = 0.001
                args.step = 1
                args.beta = 0.9
                args.weight_decay = 0.
                args.momentum = 0.999
                args.rst = 0.01
                args.ap = 0.92
                args.optimizer = 'Adam'
            elif args.dataset == 'cifar100':
                args.lr = 0.001
                args.step = 1
                args.beta = 0.9
                args.weight_decay = 0.
                args.momentum = 0.999
                args.rst = 0.01
                args.ap = 0.72
                args.optimizer = 'Adam'

            subnet = cotta.configure_model(subnet)
            params, param_names = cotta.collect_params(subnet)
            optimizer = setup_optimizer(params, args)
            adapt_model = cotta.CoTTA(subnet, optimizer,
                                      steps=args.steps,
                                      )
        elif args.algorithm == 'eata':
            # parameter setting
            if args.dataset == 'imagenet':
                args.e_margin = math.log(1000) * 0.40
                args.lr = 0.00025 if args.arch == 'Standard_R50' else 0.001
                args.fisher_alpha = 2000.
                args.d_margin = 0.05
            elif args.dataset in ['cifar10', 'cifar100']:
                args.lr = 0.005
                args.fisher_alpha = 1. if args.dataset == 'cifar10' else 2000
                args.d_margin = 0.4

                if args.dataset == 'cifar10':
                    args.e_margin = math.log(10) * 0.40
                else:
                    args.e_margin = math.log(100) * 0.40

            # compute fisher informatrix
            args.corruption = 'original'
            args.data = datasets_path[args.dataset]
            # if args.dataset=='imagenet':
            #     fisher_dataset, fisher_loader = prepare_test_data(args)
            #     fisher_dataset.set_dataset_size(args.fisher_size)
            #     fisher_dataset.switch_mode(True, False)
            # else:
            fisher_dataset, fisher_loader = load_val_dataset(args)
            subnet = eata.configure_model(subnet)
            params, param_names = eata.collect_params(subnet)
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    targets = targets.cuda()
                outputs = subnet(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in subnet.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            logger.info("compute fisher matrices finished")
            del ewc_optimizer

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = eata.EATA(subnet, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin,
                                    d_margin=args.d_margin)
        else:
            assert False, NotImplementedError
        self.adapt_model = adapt_model

        self.seed = args.seed
        # self.device = args.gpu

        self.data_path = args.data_dir
        self.dataset_name = args.dataset

        self.num_clients = len(args.common_corruptions)
        self.local_batch_nums = args.local_batches
        self.batch_size = args.batch_size

        self.args = args

    def setup(self):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # init_net(self.model, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.adapt_model.model.parameters()))})!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

        # split local dataset for each client
        local_datasets = create_datasets(self.args)

        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)

        # configure detailed settings for client upate and 
        self.setup_clients(self.args)

        # send the model skeleton to all clients
        self.transmit_model()

    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device,args=self.args)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()
        return clients

    def setup_clients(self, args):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(args)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

    def transmit_model(self):
        """Send the updated global model to selected/all clients."""

        # send the global model to all clients before the very first and after the last federated round
        # assert (self._round == 0) or (self._round == self.num_rounds)
        client_0_model = copy.deepcopy(self.clients[0].adapt_model)

        for client in tqdm(self.clients, leave=False):
            client.adapt_model = copy.deepcopy(self.adapt_model)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(
            np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices

    def average_model(self, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(self.clients)} clients...!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

        averaged_weights = OrderedDict()
        averaged_weights_ema=OrderedDict()
        for it, idx in tqdm(enumerate(list(range(len(self.clients)))), leave=False):
            local_weights = self.clients[idx].adapt_model.model.state_dict()
            if hasattr(self.adapt_model,'model_ema'):
                local_weights_ema=self.clients[idx].adapt_model.model_ema.state_dict()

            for key in self.adapt_model.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]

            if hasattr(self.adapt_model,'model_ema'):
                for key in self.adapt_model.model_ema.state_dict().keys():
                    if it == 0:
                        averaged_weights_ema[key] = coefficients[it] * local_weights_ema[key]
                    else:
                        averaged_weights_ema[key] += coefficients[it] * local_weights_ema[key]
        self.adapt_model.model.load_state_dict(averaged_weights)

        if hasattr(self.adapt_model,'model_ema'):
            self.adapt_model.model_ema.load_state_dict(averaged_weights_ema)

        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(self.clients)} clients are successfully averaged!"
        print(message);
        logging.info(message)
        del message;
        gc.collect()

    def evaluate_selected_models(self):
        """Call "client_evaluate" function of each selected client."""

        for idx in range(len(self.clients)):
            acc = self.clients[idx].client_evaluate()
            message = f"[Round: {str(self._round).zfill(4)}] ...evaluate weights of {idx} clients are successfully averaged! acc:{acc:.2f}%"
            print(message);
            logging.info(message)

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        # sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        if self.args.Federated:
            self.transmit_model()

        self.evaluate_selected_models()

        # calculate averaging coefficient of weights
        mixing_coefficients = [1. / len(self.clients) for i in range(len(self.clients))]

        # average each updated model parameters of the selected clients and update the global model
        if self.args.Federated:
            self.average_model(mixing_coefficients)


    def fit(self):
        """Execute the whole process of the federated learning."""
        # self.results = {"loss": [], "accuracy": []}
        for r in range(self.num_rounds):
            self._round = r + 1

            self.train_federated_model()

        self.transmit_model()
