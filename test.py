from main import get_args
from robustbench import load_model
from robustbench.model_zoo.enums import ThreatModel

args=get_args()


if args.arch in ['visformer_small', 'vit_base_patch16_224']:
    subnet = timm.create_model(args.arch, pretrained=True)
else:
    subnet = load_model(args.arch, args.ckpt_dir,
                        args.dataset, ThreatModel.corruptions)