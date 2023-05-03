import torch
from diffusers import StableDiffusionPipeline, DDPMPipeline


class DiffuserDemo:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_pipeline(self, type):
        if type == "stable_diffusion":
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                "sd-dreambooth-library/mr-potato-head", torch_dtype=torch.float16
            ).to(self.device)
        elif type == "butterfly":
            self.pipeline = DDPMPipeline.from_pretrained(
                "johnowhitaker/ddpm-butterflies-32px"
            ).to(self.device)

    def run(self, text=None):
        if isinstance(self.pipeline, StableDiffusionPipeline):
            images = self.pipeline(
                text, num_inference_steps=50, guidance_scale=7.5
            ).images[0]
        elif isinstance(self.pipeline, DDPMPipeline):
            images = self.pipeline(batch_size=8).images


def main():
    diffuser = DiffuserDemo()
    diffuser.create_pipeline("butterfly")
    diffuser.run()


if __name__ == "__main__":
    main()
