import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from datasets import load_dataset
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset:
    def __init__(self):
        self.dataset = load_dataset(
            "huggan/smithsonian_butterflies_subset", split="train"
        )
        self.image_size = 32
        self.batch_size = 8
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform_dataset()

    def transform_dataset(self):
        def transform(examples):
            images = [
                self.preprocess(image.convert("RGB")) for image in examples["image"]
            ]
            return {"images": images}

        self.dataset.set_transform(transform)

    def create_training_loader(self):
        self.train_dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )


class Trainer:
    def __init__(self, dataset, diffuser):
        self.epochs = 1
        self.epochs_step = 1
        self.model = diffuser.model
        self.noise_scheduler = diffuser.noise_scheduler
        self.dataset = dataset
        self.set_optimizer()

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-4)

    def sample_image_from_dataset(self, batch):
        return batch["images"].to(DEVICE)

    def sample_noise(self, clean_images):
        return torch.randn(clean_images.shape).to(DEVICE)

    def sample_timestep(self, clean_images):
        return torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (clean_images.shape[0],),
            device=DEVICE,
        ).long()

    def run_training(self):
        self.losses = []
        for epoch in range(self.epochs):
            for _, batch in enumerate(self.dataset.train_dataloader):
                clean_images = self.sample_image_from_dataset(batch)
                noise = self.sample_noise(clean_images)
                timesteps = self.sample_timestep(clean_images)

                noisy_images = self.noise_scheduler.add_noise(
                    clean_images, noise, timesteps
                )

                noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]

                loss = F.mse_loss(noise_pred, noise)
                loss.backward(loss)
                self.losses.append(loss.item())
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (epoch + 1) % self.epochs_step == 0:
                loss_last_epoch = sum(
                    self.losses[-len(self.dataset.train_dataloader) :]
                ) / len(self.dataset.train_dataloader)
                print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

    def show_learning_cruve(self):
        _, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(self.losses)
        axs[1].plot(np.log(self.losses))
        plt.show()


class CustomDiffuser:
    def __init__(self):
        self.image_size = 32

    def define_scheduler(self):
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def plot_scheduler(self):
        plt.plot(
            self.noise_scheduler.alphas_cumprod.cpu() ** 0.5,
            label=r"${\sqrt{\bar{\alpha}_t}}$",
        )
        plt.plot(
            (1 - self.noise_scheduler.alphas_cumprod.cpu()) ** 0.5,
            label=r"$\sqrt{(1 - \bar{\alpha}_t)}$",
        )
        plt.legend(fontsize="x-large")
        plt.show()

    def define_model(self):
        self.model = UNet2DModel(
            sample_size=self.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        ).to(DEVICE)


class Pipeline:
    def __init__(self, diffuser):
        self.pipeline = DDPMPipeline(
            unet=diffuser.model, scheduler=diffuser.noise_scheduler
        )

    def run_pipeline(self):
        pipeline_output = self.pipeline()
        pipeline_output.images[0].show()

    def save_pipeline(self):
        self.pipeline.save_pretrained("costum_pipeline")


def main():
    dataset = Dataset()
    dataset.transform_dataset()
    dataset.create_training_loader()

    diffuser = CustomDiffuser()
    diffuser.define_scheduler()
    diffuser.plot_scheduler()
    diffuser.define_model()

    trainer = Trainer(dataset, diffuser)
    trainer.run_training()

    pipeline = Pipeline(diffuser)
    pipeline.run_pipeline()
    pipeline.save_pipeline()


if __name__ == "__main__":
    main()
