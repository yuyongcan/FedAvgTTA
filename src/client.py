import gc
import pickle
import logging
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils.cli_utils import AverageMeter,ProgressMeter
import pickle

logger = logging.getLogger(__name__)

class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        adapt_model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device,args):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.adapt_model = None
        self.args=args
    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.adapt_model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.adapt_model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, args):
        """Set up common configuration of each client; called by center server."""
        self.batch_size=args.batch_size
        self.dataloader = DataLoader(self.data, batch_size=args.batch_size, shuffle=args.if_shuffle)

        # data_load_dir=os.path.join(args.dataloder_path,args.dataset)
        # data_load_path=os.path.join(args.dataloder_path,args.dataset,str(self.id)+'.pkl')
        # if os.path.exists(data_load_path):
        #     self.iter_dataloader=pickle.load(data_load_path)
        # else:
        #     self.iter_dataloader = iter(self.dataloader)
        #     if not os.path.exists(data_load_dir):
        #         os.makedirs(data_load_dir)
        #     with open(data_load_path,'wb') as f:
        #         pickle.dump(self.iter_dataloader,f)

        self.iter_dataloader=iter(self.dataloader)
        self.batches_now=0
        self.batches_per_round=args.local_batches
        self.top1 = AverageMeter('Acc@1', ':6.2f')
        # self.top5 = AverageMeter('Acc@5', ':6.2f')`


    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.adapt_model.to(self.device)

        # correct=0
        for i in range(self.batches_per_round):
            with torch.no_grad():
                try:
                    data, labels = next(self.iter_dataloader)
                    data, labels = data.float().to(self.device), labels.long().to(self.device)
                    outputs = self.adapt_model(data)

                    predicted = outputs.argmax(dim=1, keepdim=True)
                    correct = predicted.eq(labels.view_as(predicted)).sum().item()
                    self.top1.update(100. * correct / self.batch_size, data.size(0))
                except StopIteration:
                    message=f"client {self.id} finished adaptation"
                    print(message)
                    logging.info(message)
                    if self.device == "cuda": torch.cuda.empty_cache()
                    break
                if self.device == "cuda": torch.cuda.empty_cache()
        self.adapt_model.to('cpu')

        return self.top1.avg
