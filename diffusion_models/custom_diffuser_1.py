import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from models import UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset:
    def __init__(self):
        self.batch_size = 128
        self.dataset = torchvision.datasets.MNIST(
            root="mnist/",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

    def create_training_loader(self):
        self.train_dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

    def show_sample_data(self):
        x, y = next(iter(self.train_dataloader))
        x = x[:8]
        print("Input shape:", x.shape)
        print("Labels:", y)
        plt.imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")
        plt.show()


class CustomDiffuser:
    def __init__(self):
        self.image_size = 32

    def define_scheduler(self):
        self.noise_scheduler = NoiseScheduler()

    def define_model(self):
        self.model = UNet().to(DEVICE)

    def show_sample_diffuser(self, dataset):
        x, _ = next(iter(dataset.train_dataloader))
        x = x[:8]
        _, axs = plt.subplots(2, 1, figsize=(12, 5))
        axs[0].set_title("Input data")
        axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")

        noise = torch.rand_like(x)
        noised_x = self.noise_scheduler.add_noise(x, noise)
        axs[1].set_title("Corrupted data (-- amount increases -->)")
        axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap="Greys")
        plt.show()

    def show_model_info(self):
        x = torch.rand(8, 1, 28, 28)
        print("Model output shape : ", self.model(x).shape)
        print(
            "Number of parameters : ", sum([p.numel() for p in self.model.parameters()])
        )


class NoiseScheduler:
    def __init__(self):
        pass

    def add_noise(self, x, noise):
        amount = torch.linspace(0, 1, x.shape[0]) # TODO this is called beta
        amount = amount.view(-1, 1, 1, 1)
        return x * (1 - amount) + noise * amount


class Trainer:
    def __init__(self, dataset, diffuser):
        self.epochs = 1
        self.epochs_step = 1
        self.model = diffuser.model
        self.noise_scheduler = diffuser.noise_scheduler
        self.dataset = dataset
        self.set_optimizer()

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=4e-3)

    def sample_image_from_dataset(self, x):
        return x.to(DEVICE)

    def sample_noise(self, x):
        return torch.rand_like(x)

    def sample_timestep(self, clean_images):
        return torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (clean_images.shape[0],),
            device=DEVICE,
        ).long()

    def run_training(self):
        loss_fn = nn.MSELoss()
        self.losses = []
        for epoch in range(self.epochs):
            for x, y in self.dataset.train_dataloader:
                x = self.sample_image_from_dataset(x)
                noise = self.sample_noise(x)

                noisy_x = self.noise_scheduler.add_noise(x, noise)

                pred = self.model(noisy_x)

                loss = loss_fn(pred, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())

            if (epoch + 1) % self.epochs_step == 0:
                loss_last_epoch = sum(
                    self.losses[-len(self.dataset.train_dataloader) :]
                ) / len(self.dataset.train_dataloader)
                print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

    def show_learning_cruve(self):
        plt.plot(self.losses)
        plt.ylim(0, 0.1)
        plt.show()


def run_testing(dataset, diffuser):
    x, _ = next(iter(dataset.train_dataloader))
    x = x[:8]
    noise = torch.rand_like(x)
    noised_x = diffuser.noise_scheduler.add_noise(x, noise)

    with torch.no_grad():
        preds = diffuser.model(noised_x.to(DEVICE)).detach().cpu()

    _, axs = plt.subplots(3, 1, figsize=(12, 7))
    axs[0].set_title("Input data")
    axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap="Greys")
    axs[1].set_title("Corrupted data")
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap="Greys")
    axs[2].set_title("Network Predictions")
    axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap="Greys")
    plt.show()


def main():
    dataset = Dataset()
    dataset.create_training_loader()
    # dataset.show_sample_data()

    diffuser = CustomDiffuser()
    diffuser.define_scheduler()
    diffuser.define_model()
    # diffuser.show_sample_diffuser(dataset)
    # diffuser.show_model_info()

    trainer = Trainer(dataset, diffuser)
    trainer.run_training()

    # run_testing(dataset, diffuser)


if __name__ == "__main__":
    main()
