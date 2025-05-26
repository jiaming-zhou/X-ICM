import os
import sys
import PIL
import torch
import json
import cv2
import numpy as np
import logging
import textwrap
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
import multiprocessing as mp
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

import pickle
import cv2
import shutil
from main import PROJECT_ROOT

def center_crop_resize(image, H, W):
    W_img, H_img = image.size
    target_aspect = W / H
    input_aspect = W_img / H_img

    if input_aspect > target_aspect:
        new_width = int(target_aspect * H_img)
        new_height = H_img
    else:
        new_width = W_img
        new_height = int(W_img / target_aspect)

    left = (W_img - new_width) / 2
    top = (H_img - new_height) / 2
    right = (W_img + new_width) / 2
    bottom = (H_img + new_height) / 2

    image = image.crop((left, top, right, bottom))
    image = image.resize((W, H))

    return image


def load_image(image_path, H, W):
    image = PIL.Image.open(image_path)
    return np.array(center_crop_resize(image, H, W))[..., :3]


def load_depth_image(image_path, H, W):
    depth_image_path = image_path
    if os.path.exists(depth_image_path):
        depth_image = PIL.Image.open(depth_image_path)
    else:
        depth_image = PIL.Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    return np.array(center_crop_resize(depth_image, H, W))[..., :3]



def load_pipeline(ckpt_folder, model_id="stabilityai/stable-diffusion-2", device_id=0, seed=42):
    ckpt_folder = ckpt_folder.rstrip("/")
    device = f"cuda:{device_id}"

    # ==== Model Configuration ====
    if os.path.exists(ckpt_folder):
        folder_list = os.listdir(ckpt_folder)
        if "unet" in folder_list:
            latest_checkpoint = ckpt_folder
        else:
            checkpoint_files = [os.path.join(ckpt_folder, f) for f in folder_list if f.startswith("checkpoint")]
            latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split("-")[-1]))[-1]
        run_id = latest_checkpoint.split("/")[-3]
        run_id = os.path.join(run_id, "results", os.path.basename(latest_checkpoint))
    else:
        latest_checkpoint = ckpt_folder
        run_id = latest_checkpoint.split("/")[-1]
        run_id = os.path.join(run_id, "results")
    print(f"Loading checkpoint from {latest_checkpoint}")

    # ==== Load model ====
    unet = UNet2DConditionModel.from_pretrained(latest_checkpoint, subfolder="unet")
    #source https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, unet=unet, torch_dtype=torch.float32, use_safetensors=True
    ).to(device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # ==== Custom pipeline functions ====
    def my_prepare_image_latents(
        image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        image = image.to(device=device, dtype=dtype)
        batch_size = batch_size * num_images_per_prompt
        if image.shape[1] == 6:
            rgb, d = torch.split(image, 3, dim=1)
            image_embeds_rgb = pipe.vae.encode(rgb).latent_dist.mode()
            image_embeds_d = pipe.vae.encode(d).latent_dist.mode()
            image_latents = torch.cat([image_embeds_rgb, image_embeds_d], dim=1)
        elif image.shape[1] == 3:
            image_embeds = pipe.vae.encode(image).latent_dist.mode()
            image_latents = torch.cat([image_embeds], dim=1)
        else:
            raise ValueError("Invalid input shape")

        image_latents = torch.cat([image_latents], dim=0)
        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
        return image_latents

    pipe.set_progress_bar_config(leave=False, desc="Inference")
    pipe.prepare_image_latents = my_prepare_image_latents

    return pipe, generator, run_id


def decode_one_latent(pipe, latent, output_type="np"):
    pred_image = pipe.vae.decode(
        latent.unsqueeze(0) / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    pred_image = pred_image / 2 + 0.5
    pred_image = (
        pipe.image_processor.postprocess(
            pred_image.detach(),
            output_type=output_type,
            do_denormalize=[False],
        )
        * 255
    )
    return pred_image


### if you want to load stabilityai/stable-diffusion-2 from local file
### TODO: download from huggingface, change the model_path
model_path="/remote-home/jiamingz/projects/huggingface/hub/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79"

### load model from huggingface if not loaded from local file
if not os.path.exists(model_path):
    model_path="stabilityai/stable-diffusion-2"


ckpt_folder=os.path.join(PROJECT_ROOT, "data/dynamics_diffusion/checkpoint-20000")

pipe, generator, run_id = load_pipeline(ckpt_folder, model_id=model_path, device_id=0, seed=123)


def extract_diffusion_features(base_image_path, prompt):

    num_inference_steps = 50
    image_guidance_scale = 2.5
    guidance_scale = 2.5

    H, W = 128, 128
    image = load_image(base_image_path, H, W)
    
    depth = None
    input_image = image

    prompt_embeds = pipe._encode_prompt(
        prompt,
        device=pipe._execution_device,
        num_images_per_prompt = 1,
        do_classifier_free_guidance = True
    )
    prompt_embeds_saved = prompt_embeds[0].detach().cpu().numpy()

    input_image_copy = pipe.image_processor.preprocess(input_image / 255.0)
    input_image_latents = pipe.prepare_image_latents(
        input_image_copy,
        batch_size = 1,
        num_images_per_prompt = 1,
        dtype=prompt_embeds.dtype, ### torch.float32
        device=pipe._execution_device,
        do_classifier_free_guidance  = True,
    )
    input_image_latents_saved = input_image_latents[0].detach().cpu().numpy()


    # ==== Inference ====
    pipe.vae.config.latent_channels = 4
    edited_image = pipe(
        prompt,
        image=input_image / 255.0,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="latent",
    ).images[0]
    output_image_latents_saved = edited_image.detach().cpu().numpy()

    # ==== Decode edited image ====
    pipe.vae.config.latent_channels = 4
    latent_rgb = edited_image[:4]
    pred_image = decode_one_latent(pipe, latent_rgb)
    pred_image = pred_image.reshape(H, W, 3)
    pred_depth = None

    # === Load GT ===
    target = np.zeros((H, W, 3), dtype=np.uint8)
    target_depth =  None

    return input_image_latents_saved.flatten(), output_image_latents_saved.flatten(), prompt_embeds_saved[0,:]




if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")


