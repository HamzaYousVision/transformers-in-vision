import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader

from models import SimpleUnet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset:
    def __init__(self):
        self.batch_size = 16
        self.img_size = 32

    def load_dataset_mnist(self):
        self.dataset = torchvision.datasets.MNIST(
            root="mnist/",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

    def load_dataset_cifar(self):
        self.dataset = torchvision.datasets.CIFAR10(
            root="cifar/",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
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
        train = torchvision.datasets.MNIST(
            root="mnist/", download=True, transform=data_transform
        )
        test = torchvision.datasets.MNIST(
            root="mnist/", download=True, transform=data_transform, split="test"
        )
        self.dataset = torch.utils.data.ConcatDataset([train, test])

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
        plt.imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")
        plt.show()


class NoiseScheduler:
    def __init__(self):
        self.T = 10  #  300
        self.precalculate_terms()

    def precalculate_terms(self):
        betas = self.linear_beta_schedule(timesteps=self.T)
        alphas = torch.cumprod(1.0 - betas, axis=0)
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_one_alphas = torch.sqrt(1.0 - alphas)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(DEVICE)

    def add_noise(self, x, noise, t):
        sqrt_alphas = self.get_index_from_list(self.sqrt_alphas, t, x.shape).to(DEVICE)
        sqrt_one_alphas = self.get_index_from_list(self.sqrt_one_alphas, t, x.shape).to(
            DEVICE
        )
        return sqrt_alphas * x + sqrt_one_alphas * noise


class CustomDiffuser:
    def __init__(self):
        pass

    def define_scheduler(self):
        self.noise_scheduler = NoiseScheduler()

    def define_model(self):
        self.model = SimpleUnet()

    def show_sample_diffuser(self, dataset):
        x, _ = next(iter(dataset.train_dataloader))
        x = x[:8]
        _, axs = plt.subplots(2, 1, figsize=(12, 5))
        axs[0].set_title("Input data")
        axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")

        amount = torch.linspace(0, 1, x.shape[0])
        noised_x = self.noise_scheduler.add_noise(x, amount)
        axs[1].set_title("Corrupted data (-- amount increases -->)")
        axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap="Greys")
        plt.show()
    
    def show_sample_image_diffuser(self, dataset):
        image = next(iter(dataset.train_dataloader))[0]
        plt.figure(figsize=(15, 15))
        plt.axis("off")

        num_images = 10
        T = self.noise_scheduler.T
        stepsize = int(T / num_images)
        for idx in range(0, T, stepsize): 
            t = torch.Tensor([idx]).type(torch.int64)
            plt.subplot(1, num_images + 1, int(idx / stepsize) + 1)
            noise = torch.rand(image.shape[0]).to(DEVICE)
            img  = self.noise_scheduler.add_noise(image, noise, t)
            self.dataset.show_tensor_image(img)
        plt.show()

    def show_model_info(self):
        x = torch.rand(8, 1, 28, 28)
        print("Model output shape : ", self.model(x).shape)
        print(
            "Number of parameters : ", sum([p.numel() for p in self.model.parameters()])
        )


class Trainer:
    def __init__(self, dataset, diffuser):
        self.epochs = 3
        self.model = diffuser.model.to(DEVICE)
        self.noise_scheduler = diffuser.noise_scheduler
        self.dataset = dataset
        self.T = diffuser.noise_scheduler.T
        self.batch_size = dataset.batch_size
        self.set_optimizer()

    def set_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def sample_image_from_dataset(self, batch):
        return batch[0]

    def sample_timestep(self):
        return torch.randint(0, self.T, (self.batch_size,), device=DEVICE).long()

    def sample_noise(self, x):
        return torch.randn_like(x).to(DEVICE)

    def run(self):
        for epoch in range(self.epochs):
            for step, batch in enumerate(self.dataset.train_dataloader):
                t = self.sample_timestep()
                x = self.sample_image_from_dataset(batch)
                noise = self.sample_noise(x)

                x_noisy = self.noise_scheduler.add_noise(x, noise, t)

                noise_pred = self.model(x_noisy, t)

                loss = F.l1_loss(noise, noise_pred)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if epoch % 1 == 0 and step == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: loss.item() ")


def main():
    dataset = Dataset()
    dataset.load_dataset_cifar()
    dataset.create_training_loader()

    diffuser = CustomDiffuser()
    diffuser.define_scheduler()
    diffuser.define_model()

    trainer = Trainer(dataset, diffuser)
    trainer.run()


if __name__ == "__main__":
    main()
