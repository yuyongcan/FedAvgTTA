import gc
import pickle
import logging

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils.cli_utils import AverageMeter,ProgressMeter

logger = logging.getLogger(__name__)

def adapt_model_to(adapt_model, device):
    adapt_model.model.to(device)
    if hasattr(adapt_model,'model_ema'):
        adapt_model.model_ema.to(device)
    if hasattr(adapt_model,'model_anchor'):
        adapt_model.model_anchor.to(device)

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
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.adapt_model = None

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
        self.iter_dataloader=iter(self.dataloader)
        self.batches_now=0
        self.batches_per_round=args.local_batches
        self.top1 = AverageMeter('Acc@1', ':6.2f')
        # self.top5 = AverageMeter('Acc@5', ':6.2f')


    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        adapt_model_to(self.adapt_model,self.device)

        # correct=0
        for i in range(self.batches_per_round):
            with torch.no_grad():
                try:
                    data, labels = next(self.iter_dataloader)
                    data, labels = data.float().to("cuda:" + self.device), labels.long().to(self.device)
                    outputs = self.adapt_model(data)

                    predicted = outputs.argmax(dim=1, keepdim=True)
                    correct = predicted.eq(labels.view_as(predicted)).sum().item()
                    self.top1.update(100. * correct / self.batch_size, data.size(0))
                except StopIteration:
                    print(f"client {self.id} finished adaptation")
                    if self.device == "cuda": torch.cuda.empty_cache()
                    break
                if self.device == "cuda": torch.cuda.empty_cache()
        adapt_model_to(self.adapt_model,'cpu')

        return self.top1.avg
