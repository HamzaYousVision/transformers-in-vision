
import argparse

from torchvision.models import vit_b_16, resnet50, swin_b
from model_structure import show_model_structure


def main(args):
    print(args.model)
    if args.model == "vit": 
        model = vit_b_16()
    elif args.model == "resnet":
        model = resnet50()
    elif args.model == "swin":
        model = swin_b() 
    else: 
        SystemExit("Not Valid")
    show_model_structure(model)

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description="Model Structure Understanding Script")
    parser.add_argument("--model", type=str, help="model name")
    args = parser.parse_args()
    
    main(args)