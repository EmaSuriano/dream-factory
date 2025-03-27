from diffusers import DiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import UnCLIPPipeline
from diffusers import StableDiffusionPipeline

import torch


def main():
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("output.png")


if __name__ == "__main__":
    main()
