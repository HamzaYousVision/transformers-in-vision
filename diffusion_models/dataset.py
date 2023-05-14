

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets
from torchvision.utils import make_grid


class Dataset:
    def __init__(self):
        self.batch_size = 16
        self.img_size = 32

    def load_dataset(self, type):
        if "cifar" in type: 
            self.load_dataset_cifar()
        elif "mnist" in type:
            self.load_dataset_mnist()

    def load_dataset_mnist(self):
        self.dataset = datasets.MNIST(
            root="mnist/",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

    def load_dataset_cifar(self):
        self.dataset = datasets.CIFAR10(
            root="cifar/",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

    def load_transformed_dataset(self):
        data_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )
        train = datasets.MNIST(
            root="mnist/", download=True, transform=data_transform
        )
        test = datasets.MNIST(
            root="mnist/", download=True, transform=data_transform, split="test"
        )
        self.dataset = ConcatDataset([train, test])

    def show_tensor_image(self, image):
        reverse_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                transforms.Lambda(lambda t: t * 255.0),
                transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
                transforms.ToPILImage(),
            ]
        )
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        plt.imshow(reverse_transforms(image))
        # plt.show()

    def create_training_loader(self):
        picks = np.random.permutation(16)
        self.train_dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=picks
        )

    def show_sample_data(self):
        x, y = next(iter(self.train_dataloader))
        x = x[:8]
        print("Input shape:", x.shape)
        print("Labels:", y)
        plt.imshow(make_grid(x)[0], cmap="Greys")
        plt.show()
