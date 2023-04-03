import torch
import torch.nn as nn

from methods.tent import softmax_entropy


class T3A(nn.Module):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, hparams,model):
        super().__init__()
        self.hparams = hparams
        modules= list(model.children())[:-1]
        self.featurizer = nn.Sequential(*modules)
        self.classifier = list(model.children())[-1]
        num_classes = self.classifier.out_features

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports.detach()
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob).detach()
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes).float().detach()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = hparams['filter_K']
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=False):
        if not self.hparams['cached_loader']:
            z = self.featurizer(x).squeeze()
        else:
            z = x
        if adapt:
            # online adaptation
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z.detach()])
            self.labels = torch.cat([self.labels, yhat.detach()])
            self.ent = torch.cat([self.ent, ent.detach()])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        # if filter_K == -1:
        #     indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    def transmit(self,adapt_model):
        self.supports=adapt_model.supports
        self.labels=adapt_model.labels
        self.ent=adapt_model.ent