import torch
import torchextractor as tx

class FeatureExtractor: 
    def __init__(self, model):
        self.model = model

    def extract_features(self, image_input=None): 
        layers_names = [name for name, _ in self.model.named_modules()]
        model_tx = tx.Extractor(self.model, layers_names)

        if image_input is None: 
            input_activation_size = torch.Size((1, 3, 224, 224)) 
            image_input = torch.rand(input_activation_size)

        _, features = model_tx(image_input)
        activation_shapes = {name: f for name, f in features.items()}

        activation_shapes = {}
        for name, f in features.items():
            if type(f) == tuple:
                activation_shapes[name] = f[0].shape
            else: 
                activation_shapes[name] = f.shape 

