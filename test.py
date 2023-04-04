# args=get_args()
#
# if args.dataset == 'imagenet':
#     # args.arch = 'Standard_R50'
#     args.data_dir = '/data/yongcan.yu/datasets'
#
# elif args.dataset in ['cifar10', 'cifar100']:
#     args.data_dir = './data'
#     if args.dataset == 'cifar10':
#         args.arch = 'Standard'
#     elif args.dataset == 'cifar100':
#         args.arch = 'Hendrycks2020AugMix_ResNeXt'
#
# if args.arch in ['visformer_small', 'vit_base_patch16_224']:
#     subnet = timm.create_model(args.arch, pretrained=True)
# else:
#     subnet = load_model(args.arch, args.ckpt_dir,
#                         args.dataset, ThreatModel.corruptions)
# print([name for name, param in subnet.named_parameters()])
# trainset = torchvision.datasets.ImageNet(root=datasets_path['imagenet'], split='train',
#                                                 transform=te_transforms)
# length= len(trainset)
# idx = torch.randperm(length)[:150000]
# trainset = torch.utils.data.Subset(trainset, idx)

a = [1, 2, 3]
b = a[0:1]
print(b)
