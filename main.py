import argparse
from matplotlib import pyplot as plt

from torchvision.models import vit_b_16, resnet50, swin_b
from feature_extraction import FeatureExtractor
from model_structure import show_model_structure
from attention_visualization import AttentionVisualization


def main(args):
    if args.model == "vit":
        model = vit_b_16()
    elif args.model == "resnet":
        model = resnet50()
    elif args.model == "swin":
        model = swin_b()
    else:
        SystemExit("Not Valid")

    if args.structure:
        show_model_structure(model)
    if args.attention:
        attention_visualization = AttentionVisualization(model)
        attention_visualization.visualize()
    if args.features:
        features_extractor = FeatureExtractor(model)
        features_extractor.set_input_image()
        features_extractor.extract_features()
        features_extractor.show_info()
        features_extractor.visualize_features(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Structure Understanding Script")
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--features", action="store_true", help="feature extraction")
    parser.add_argument("--attention", action="store_true", help="visualize attention")
    parser.add_argument("--structure", action="store_true", help="show model structure")
    args = parser.parse_args()

    main(args)
