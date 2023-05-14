import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from matplotlib import pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, dataset, diffuser):
        self.epochs = 1
        self.epochs_step = 1
        self.model = diffuser.model.to(DEVICE)
        self.noise_scheduler = diffuser.noise_scheduler
        self.dataset = dataset
        self.batch_size = dataset.batch_size
        self.set_optimizer()

    def set_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def sample_image_from_dataset(self, batch):
        return batch[0]

    def sample_timestep(self):
        return torch.randint(0, self.noise_scheduler.timesteps, (self.batch_size,), device=DEVICE).long()

    def sample_noise(self, x):
        return torch.randn_like(x).to(DEVICE)

    def run(self):
        self.losses = []
        for epoch in range(self.epochs):
            for batch in self.dataset.train_dataloader:
                t = self.sample_timestep()
                x = self.sample_image_from_dataset(batch)
                noise = self.sample_noise(x)

                x_noisy = self.noise_scheduler.add_noise(x, noise, t)

                noise_pred = self.model(x_noisy, t)

                loss = F.l1_loss(noise, noise_pred)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())

                if (epoch + 1) % self.epochs_step == 0:
                    loss_last_epoch = sum(
                        self.losses[-len(self.dataset.train_dataloader) :]
                    ) / len(self.dataset.train_dataloader)
                    print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")


class SimplifiedTrainer(Trainer):
    def __init__(self, dataset, diffuser):
        super().__init__(dataset, diffuser)

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

    def run(self):
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

