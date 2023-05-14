from dataset import Dataset
from diffuser import CustomDiffuser 
from trainer import Trainer

def main():
    dataset = Dataset ()
    dataset.load_dataset("cifar")
    dataset.create_training_loader()

    diffuser = CustomDiffuser()
    diffuser.define_scheduler()
    diffuser.define_model()

    trainer = Trainer(dataset, diffuser)
    trainer.run() 

if __name__ == "__main__":
    main()

