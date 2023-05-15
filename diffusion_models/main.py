import argparse
from dataset import Dataset
from diffuser import Diffuser, SimplifiedDiffuser 
from trainer import Trainer, SimplifiedTrainer

def create_dataset(simplified=False): 
    dataset = Dataset()
    if simplified: 
        dataset.load_dataset('mnist')
    else:
        dataset.load_dataset("cifar")
    dataset.create_training_loader()
    return dataset

def create_diffuser(simplified=False):
    if simplified:
        diffuser = SimplifiedDiffuser()
    else: 
        diffuser = Diffuser()
    diffuser.define_scheduler()
    diffuser.define_model()
    return diffuser

def run_training(dataset, diffuser, simplified=False):
    if simplified:
        trainer = SimplifiedTrainer(dataset, diffuser)
    else: 
        trainer = Trainer(dataset, diffuser)
    trainer.run() 


def main(args):
    dataset = create_dataset(args.simplified)
    diffuser = create_diffuser(args.simplified)
    run_training(dataset, diffuser, args.simplified)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion model script")
    parser.add_argument("--simplified", action="store_true", help="simplified implementation")
    args = parser.parse_args()

    main(args)
