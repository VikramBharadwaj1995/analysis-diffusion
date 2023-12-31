# Darshil Patel
# December 14, 2022
# CS7180 Advanced Perception
# This script contains code for adding a new style to a Stable Diffusion model
# and visually evaluating training and results of image generation using 
# DAAMs (Diffusion Attentive Attribution Maps)

import argparse
import itertools
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import daam
from daam import trace
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import accelerate

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def image_grid(imgs, rows, cols):
    """
      Plots images in a grid to display them side by side.
      :param imgs: images to put on gird
      :param rows: number of images per row
      :param rows: number of images per column
      :return: the image
    """
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


"""## Setup for adding a new concept"""

# Path for our pre-trained stable diffusion model
pretrained_model_name_or_path = "stabilityai/stable-diffsion-2"

# URLs for the images representing our new style 
urls = [
    "https://www.alexgrey.com/art-images/Psychedelic-Healing---2020-Alex-Grey-smaller-watermarked.jpg",
    "https://www.alexgrey.com/art-images/Godself-2012-Alex-Grey-watermarked.jpeg",
    "https://www.alexgrey.com/art-images/Rainbow-Eye-Ripple-2019-Alex-Grey-Allyson-Grey-watermarked.jpeg"]

# Display the training images
import requests
import glob
from io import BytesIO


def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


images = list(filter(None, [download_image(url) for url in urls]))
save_path = "./my_concept"
if not os.path.exists(save_path):
    os.mkdir(save_path)
[image.save(f"{save_path}/{i}.jpeg") for i, image in enumerate(images)]
image_grid(images, 1, len(images))

# Setup tokens and parameters for teaching the model
# Here we set up the tokens that we use to train the concept in the model
what_to_teach = "style"
placeholder_token = "<ag-e>"
initializer_token = "ag"

# Training the model using the textual inversion method

# Create Dataset

# These prompts are sampled from Google ImageGen templates and are suitable to 
# help teach the new concept in the text embedding space
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


# Setup the dataset
class TextualInversionDataset(Dataset):
    def __init__(
            self,
            data_root,
            tokenizer,
            learnable_property="object",  # [object, style]
            size=512,
            repeats=100,
            interpolation="bicubic",
            flip_p=0.5,
            set="train",
            placeholder_token="*",
            center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


# Setup the model

# Load the tokenizer and add the placeholder token as a additional special token.
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)

# Add the placeholder token in tokenizer
num_added_tokens = tokenizer.add_tokens(placeholder_token)
if num_added_tokens == 0:
    raise ValueError(
        f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
        " `placeholder_token` that is not already in the tokenizer."
    )

# Get token ids for our placeholder and initializer token
# Convert the initializer_token, placeholder_token to ids
token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
if len(token_ids) > 1:
    raise ValueError("The initializer token must be a single token.")

initializer_token_id = token_ids[0]
placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

# Load the Stable Diffusion model, here we obtain the main pre-trained components of the diffusion model
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)

# We have added the placeholder_token in the tokenizer so we resize the 
# token embeddings here this will a new embedding vector in the token embeddings
# for our placeholder_token
text_encoder.resize_token_embeddings(len(tokenizer))


# Initialize the newly added placeholder token with the embeddings of the initializer token
token_embeds = text_encoder.get_input_embeddings().weight.data
token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

# Freeze rest of the model parameters here since we are only training the text
# encoder
def freeze_params(params):
    """
    Freezes the parameters in the models so the weights are not changed while
    training
    """
    for param in params:
        param.requires_grad = False


# Freeze vae and unet and encoder parameters
freeze_params(vae.parameters())
freeze_params(unet.parameters())
params_to_freeze = itertools.chain(
    text_encoder.text_model.encoder.parameters(),
    text_encoder.text_model.final_layer_norm.parameters(),
    text_encoder.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)

# Create our training data
train_dataset = TextualInversionDataset(
    data_root=save_path,
    tokenizer=tokenizer,
    size=vae.sample_size,
    placeholder_token=placeholder_token,
    repeats=100,
    learnable_property=what_to_teach,
    center_crop=False,
    set="train",
)


def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)


# Create noise_scheduler for training
noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

# Training
# Define hyperparameters for our training
# Setting up all training args
hyperparameters = {
    "learning_rate": 1e-06,
    "scale_lr": True,
    "max_train_steps": 1200,
    "save_steps": 250,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "output_dir": "new-concept"
}

# Training function
logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
    """
    Shows the progress and what step in training the model is in.
    """
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


def training_function(text_encoder, vae, unet):
    """
    Runs training on the diffusion model. The text encoder is the main component
    being trained here. We go through multiple iterations while using prompts
    that encode the new concept being learned to guide the model to associate
    the visual markers in the training images to the text in the prompt.
    """
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]
    gradient_checkpointing = hyperparameters["gradient_checkpointing"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=hyperparameters["mixed_precision"],
    )

    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    train_dataloader = create_dataloader(train_batch_size)

    if hyperparameters["scale_lr"]:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    # weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    # weight_dtype = torch.bfloat16

    # Move vae and unet to device
    # vae.to(accelerator.device, dtype=weight_dtype)
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)
    unet.to(accelerator.device)

    # Keep vae in eval mode
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    # Set seeding for the training and daam generation
    set_seed(0)
    torch.manual_seed(0)
    gen = daam.set_seed(0)

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                # latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                # noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % hyperparameters["save_steps"] == 0:
                    save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            # Every 10 global iterations we observe the attribution map for the new object being trained
            if global_step % 50 == 0 or global_step <= 5:
                # Set up pipeline for text to image inference
                pipe = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    vae=vae,
                    unet=unet,
                )
                pipe.save_pretrained(output_dir)
                # Also save the newly trained embeddings
                save_path = os.path.join(output_dir, f"learned_embeds.bin")
                save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

                # Display and save DAAM
                pipe.to("cuda")
                file_name = "images/" + initializer_token + str(global_step) + ".png"
                prompt = "<ag-e>"
                with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
                    with trace(pipe) as tc:
                        out = pipe(prompt, num_inference_steps=30, generator=gen)
                        heat_map = tc.compute_global_heat_map(prompt)
                        heat_map = heat_map.compute_word_heat_map("<ag-e>")
                        heat_map.plot_overlay(out.images[0])
                        # Show image
                        plt.title(placeholder_token)
                        plt.savefig(file_name)
                        plt.show()
                del pipe

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline and save the model
    if accelerator.is_main_process:
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipe.save_pretrained(output_dir)
        # Also save the newly trained embeddings
        save_path = os.path.join(output_dir, f"learned_embeds.bin")
        save_progress(text_encoder, placeholder_token_id, accelerator, save_path)


import accelerate

accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet))

for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
    if param.grad is not None:
        del param.grad  # free some memory
    torch.cuda.empty_cache()

# Testing the Trained Model with the New Style

# Set up the model pipeline so we can input prompts for image generation
pipe = StableDiffusionPipeline.from_pretrained(
    hyperparameters["output_dir"],
    # "downloaded_embedding",
    torch_dtype=torch.float16,
    force_download=True,
).to("cuda")

# Run the Stable Diffusion pipeline
prompt = "Tree in the style of <ag-e>"  # @param {type:"string"}
num_samples = 2
num_rows = 1

all_images = []
for _ in range(num_rows):
    images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_rows, num_samples)

# Display Generated Image with the Heatmap for Style Association
prompt = "tree in the style of <ag-e>"
gen = daam.set_seed(0)
with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=30, generator=gen)
        heat_map = tc.compute_global_heat_map(prompt)
        heat_map = heat_map.compute_word_heat_map("<ag-e>")
        heat_map.plot_overlay(out.images[0])
        # Show image
        plt.title(placeholder_token)
        plt.show()
