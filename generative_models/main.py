import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from six.moves import xrange
import umap

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from vq_vae import VQVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data:
    def __init__(self):
        self.define_training_loader()
        self.define_validation_loader()

    def define_training_loader(self):
        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                ]
            ),
        )
        self.training_loader = DataLoader(
            training_data,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )
        self.data_variance = np.var(training_data.data / 255.0)

    def define_validation_loader(self):
        validation_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                ]
            ),
        )
        self.validation_loader = DataLoader(
            validation_data, batch_size=32, shuffle=True, pin_memory=True
        )


class Trainer:
    def __init__(self, model, data_loader):
        self.model = model.model
        self.data_loader = data_loader

        self.num_training_updates = 200
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, amsgrad=False)

        self.model.train()
        self.train_res_recon_error = []
        self.train_res_perplexity = []

    def train(self):
        print("\n training ...")
        for i in xrange(self.num_training_updates):
            (data, _) = next(iter(self.data_loader.training_loader))
            data = data.to(device)
            self.optimizer.zero_grad()

            vq_loss, data_recon, perplexity = self.model(data)
            recon_error = F.mse_loss(data_recon, data) / self.data_loader.data_variance
            loss = recon_error + vq_loss
            loss.backward()

            self.optimizer.step()

            self.train_res_recon_error.append(recon_error.item())
            self.train_res_perplexity.append(perplexity.item())

            if (i + 1) % 10 == 0:
                print("%d iterations" % (i + 1))
                print("recon_error: %.3f" % np.mean(self.train_res_recon_error[-100:]))
                print("perplexity: %.3f" % np.mean(self.train_res_perplexity[-100:]))
                print()

    def plot_training(self):
        train_res_recon_error_smooth = savgol_filter(self.train_res_recon_error, 11, 7)
        train_res_perplexity_smooth = savgol_filter(self.train_res_perplexity, 11, 7)

        f = plt.figure(figsize=(16, 8))
        ax = f.add_subplot(1, 2, 1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale("log")
        ax.set_title("Smoothed NMSE.")
        ax.set_xlabel("iteration")

        ax = f.add_subplot(1, 2, 2)
        ax.plot(train_res_perplexity_smooth)
        ax.set_title("Smoothed Average codebook usage (perplexity).")
        ax.set_xlabel("iteration")
        ax.show()


class Validator:
    def __init__(self, model, data_loader):
        self.model = model.model
        self.data_loader = data_loader

    def validate(self):
        self.model.eval()

        (valid_originals, _) = next(iter(self.data_loader.validation_loader))
        valid_originals = valid_originals.to(device)

        vq_output_eval = self.model._pre_vq_conv(self.model._encoder(valid_originals))
        _, valid_quantize, _, _ = self.model._vq_vae(vq_output_eval)
        valid_reconstructions = self.model._decoder(valid_quantize)

        (train_originals, _) = next(iter(self.data_loader.training_loader))
        train_originals = train_originals.to(device)
        _, train_reconstructions, _, _ = self.model._vq_vae(train_originals)

def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

def main():
    data_loader = Data()
    model_vq_vae = VQVAE()

    trainer = Trainer(model_vq_vae, data_loader)
    trainer.train()
    trainer.plot_training()

    validator = Validator(model_vq_vae, data_loader)
    validator.validate()

    proj = umap.UMAP(n_neighbors=3, min_dist=0.1, metric="cosine").fit_transform(
        model_vq_vae.model._vq_vae._embedding.weight.data.cpu()
    )

    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()



