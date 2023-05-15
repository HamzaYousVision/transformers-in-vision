import argparse
from transformers import pipeline
from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining
import numpy as np
import torch


def run_video_encoding():
    num_frames = 16
    video = list(np.random.uniform(0, 1, size=(16, 3, 224, 224)))

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-large")
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-large")

    pixel_values = feature_extractor(video, return_tensors="pt").pixel_values

    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

    outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
    loss = outputs.loss
    print(loss)


def run_sentiment_analysis():
    classifier = pipeline("sentiment-analysis")
    results = classifier(["I am very happy that my daughter is doing good.", "Looking forward to see my daughter at home."])
    for result in results: 
        print(result)



def main(args):
    if "sentiment" in args.task:
        run_sentiment_analysis()
    elif "video" in args.task:
        run_video_encoding()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face getting started script")
    parser.add_argument("--task", type=str, help="task")
    args = parser.parse_args()

    main(args)