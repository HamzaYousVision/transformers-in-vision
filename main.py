from torchvision.models import vit_b_16
from model_structure import ModelStructure

def main(): 
    vit = vit_b_16()
    vit_structure = ModelStructure(vit) 
    vit_structure.show_model_layers()

    


if __name__ == "__main__":
    main()