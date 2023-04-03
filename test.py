import torch
import torchvision

from main import get_args
from robustbench import load_model
from robustbench.model_zoo.enums import ThreatModel
from src.data import datasets_path, te_transforms

# args=get_args()
#
#
# if args.arch in ['visformer_small', 'vit_base_patch16_224']:
#     subnet = timm.create_model(args.arch, pretrained=True)
# else:
#     subnet = load_model(args.arch, args.ckpt_dir,
#                         args.dataset, ThreatModel.corruptions)

trainset = torchvision.datasets.ImageNet(root=datasets_path['imagenet'], split='train',
                                                transform=te_transforms)
length= len(trainset)
idx = torch.randperm(length)[:150000]
trainset = torch.utils.data.Subset(trainset, idx)
print(1)