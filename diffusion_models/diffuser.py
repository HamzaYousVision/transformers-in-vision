from typing import Any
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models import SimpleUnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseScheduler:
    def __init__(self):
        self.timesteps = 10
        self.precalculate_terms()

    def precalculate_terms(self):
        betas = self.linear_beta_schedule()
        alphas = torch.cumprod(1.0 - betas, axis=0)
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_one_alphas = torch.sqrt(1.0 - alphas)

    def linear_beta_schedule(self, start=0.0001, end=0.02):
        return torch.linspace(start, end, self.timesteps)

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
    
class SimplifiedNoiseScheduler(NoiseScheduler):
    def __init__(self):
        super().__init__()

    def add_noise(self, x, noise):
        amount = torch.linspace(0, 1, x.shape[0]) # TODO this is called beta
        amount = amount.view(-1, 1, 1, 1)
        return x * (1 - amount) + noise * amount

    
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
        axs[0].imshow(make_grid(x)[0], cmap="Greys")

        amount = torch.linspace(0, 1, x.shape[0])
        noised_x = self.noise_scheduler.add_noise(x, amount)
        axs[1].set_title("Corrupted data (-- amount increases -->)")
        axs[1].imshow(make_grid(noised_x)[0], cmap="Greys")
        plt.show()
    
    def show_sample_image_diffuser(self, dataset):
        image = next(iter(dataset.train_dataloader))[0]
        plt.figure(figsize=(15, 15))
        plt.axis("off")

        num_images = 10
        stepsize = int(self.noise_scheduler.timesteps / num_images)
        for idx in range(0, self.noise_scheduler.timesteps, stepsize): 
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