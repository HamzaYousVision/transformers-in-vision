import os
import random
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torchextractor as tx
from torchvision import transforms


class FeatureExtractor:
    def __init__(self, model):
        self.image_input = None
        self.input_size = (224, 224)
        self.model = model

    def set_input_image(self, random_input=False):
        if random_input:
            self.image_input = torch.rand(
                torch.Size((1, 3, self.input_size[0], self.input_size[1]))
            )
        else:
            self.image_input = Image.open(os.path.join("data", "sample_input.jpg"))
            self.image_input = self.generate_input(self.image_input).unsqueeze(0)

    def extract_features(self):
        assert self.image_input is not None, "please reset image first"
        layers_names = [name for name, _ in self.model.named_modules()]
        model_tx = tx.Extractor(self.model, layers_names)
        _, self.features = model_tx(self.image_input)

    def generate_input(self, input):
        transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transform(input)

    def show_info(self):
        for name, f in self.features.items():
            if type(f) == tuple:
                print(name, f[0].shape, sep="  ")
            else:
                print(name, f.shape, sep="  ")

    def visualize_features(self, features_index):
        slected_layer = list(self.features)[features_index]
        selected_feature = self.features[slected_layer]
        
        if len(selected_feature.shape) == 4:
            channel_indexes = random.sample(range(selected_feature.shape[-1]), 50)
            for channel_index in channel_indexes:
                plan = selected_feature[0, :, :, channel_index].cpu().detach().numpy()

                plt.imshow(plan)
                plt.pause(0.1)
                plt.draw
