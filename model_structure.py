from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

class ModelStructure:
    def __init__(self, model):
        self.model = model

    def classify_layer(self, layer):
        if isinstance(layer, (Conv2d, Linear, Dropout, LayerNorm)): 
            return "operation"
        else:
            return "block"


    def show_model_layers(self):
        # TODO : make recurssive
        # level 1
        for name, layer in self.model.named_children():
            print(name, type(layer))
            layer_type = self.classify_layer(layer)
            if layer_type == "block":

                # level 2
                for name_block, layer_block in layer.named_children():
                    print("...", name_block, type(layer_block))
                    subblock_type = self.classify_layer(layer_block)
                    if subblock_type == "block":

                        # level 3
                        for name_subblock, layer_subblock in layer_block.named_children():
                            print("......", name_subblock, type(layer_subblock))
                            subsubblock_type = self.classify_layer(layer_subblock)
                            if subsubblock_type == "block":

                                # level 4
                                for name_subsubblock, layer_subsubblock in layer_subblock.named_children():
                                    print(".........", name_subsubblock, type(layer_subsubblock)) 
