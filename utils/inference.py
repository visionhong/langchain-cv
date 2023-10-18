import sys
sys.path.append('Grounded-Segment-Anything')
sys.path.append('Grounded-Segment-Anything/EfficientSAM')

import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageOps

from diffusers import  AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

from utils.lama_cleaner_helper import norm_img
from utils.model_setup import get_triton_client, get_sd_inpaint, get_lama_cleaner, get_instruct_pix2pix, get_male_anime_generator, get_female_anime_generator, get_general_generator
import tritonclient.http

def instruct_pix2pix(image, prompt):
    pipe = get_instruct_pix2pix()
    
    image = ImageOps.exif_transpose(image)
    images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
    return images

def sam(image, neg_coords, pos_coords, labels):
    triton_client = get_triton_client()
    image = np.array(image).copy()
    
    pos_coords_in = tritonclient.http.InferInput("pos_coords", pos_coords.shape, "INT64")
    neg_coords_in = tritonclient.http.InferInput("neg_coords", neg_coords.shape, "INT64")
    labels_in = tritonclient.http.InferInput("labels", labels.shape, "INT64")
    image_in = tritonclient.http.InferInput("input_image", image.shape, "UINT8")

    mask_out = tritonclient.http.InferRequestedOutput(name="mask", binary_data=False)
    image_out = tritonclient.http.InferRequestedOutput(name="segmented_image", binary_data=False)
    
    pos_coords_in.set_data_from_numpy(pos_coords.astype(np.int64))
    neg_coords_in.set_data_from_numpy(neg_coords.astype(np.int64))
    labels_in.set_data_from_numpy(labels.astype(np.int64))
    image_in.set_data_from_numpy(image.astype(np.uint8))
    
    response = triton_client.infer(
        model_name="sam", model_version="1", 
        inputs=[pos_coords_in, neg_coords_in, labels_in, image_in], 
        outputs=[mask_out, image_out]
    )
    mask = response.as_numpy("mask")
    segmented_image = response.as_numpy("segmented_image")

    return image, mask, Image.fromarray(segmented_image)
 
    
def sd_inpaint(image: Image.Image, mask: Image.Image, inpaint_prompt):

    pipe = get_sd_inpaint()
    inpainted_image = pipe(prompt=inpaint_prompt, image=image, mask_image=mask, width=mask.size[0], height=mask.size[1]).images[0]
    
    return inpainted_image
   
def lama_cleaner(image, mask, device):
    model = get_lama_cleaner()
    
    image = norm_img(image)
    
    mask = norm_img(mask)

    mask = (mask > 0) * 1
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    inpainted_image = model(image, mask)

    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")

    return Image.fromarray(cur_res)


def general_generator(prompt, num_images, device, use_controlnet=False):

    if use_controlnet:
        pipe, model_name = get_general_generator(use_controlnet=True, device=device)
        
        images = pipe(
            prompt=prompt,
            guidance_scale=7,
            num_inference_steps=20,
            image = st.session_state[model_name],
            num_images_per_prompt=num_images
        ).images

        return images
    else:
        pipe, _ = get_general_generator(use_controlnet=False, device=device) 
        
        images = pipe(
            prompt=prompt,
            guidance_scale=7,
            num_inference_steps=20,
            num_images_per_prompt=num_images
        ).images

        return images
    
    
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
    # pipeline = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to(device)

    # images = pipeline(
    #     prompt, 
    #     width=1024,
    #     height=1024,
    #     prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    #     prior_guidance_scale=4.0,
    #     num_images_per_prompt=num_images,
    # ).images
    
    # return images


def male_anime_generator(prompt, num_images, device, use_controlnet=False):

    if use_controlnet:
        pipe, model_name = get_male_anime_generator(use_controlnet=True, device=device)
        
        images = pipe(
            prompt=prompt,
            negative_prompt="7dirtywords,an10,bad-picture-chill-75v,badhandv4,EasyNegative,By bad artist -neg,negative_hand Negative Embedding _negative_hand,ng_deepnegative_v1_75t,Unspeakable-Horrors-Composition-4v,verybadimagenegative_v1.3,",
            guidance_scale=7,
            num_inference_steps=20,
            image = st.session_state[model_name],
            num_images_per_prompt=num_images
        ).images

        return images
    else:
        pipe, _ = get_male_anime_generator(use_controlnet=False, device=device) 
        
        images = pipe(
            prompt=prompt,
            negative_prompt="7dirtywords,an10,bad-picture-chill-75v,badhandv4,EasyNegative,By bad artist -neg,negative_hand Negative Embedding _negative_hand,ng_deepnegative_v1_75t,Unspeakable-Horrors-Composition-4v,verybadimagenegative_v1.3,",
            guidance_scale=7,
            num_inference_steps=20,
            num_images_per_prompt=num_images
        ).images

        return images
    
    
def female_anime_generator(prompt, num_images, device, use_controlnet=False):

    if use_controlnet:
        pipe, model_name = get_female_anime_generator(use_controlnet=True, device=device)
        
        images = pipe(
            prompt=prompt,
            negative_prompt="(worst quality:2), (low quality:1.8), (normal quality:1.6), bad-hands-5,",
            guidance_scale=7,
            num_inference_steps=40,
            image = st.session_state[model_name],
            num_images_per_prompt=num_images
        ).images

        return images
    else:
        pipe, _ = get_female_anime_generator(use_controlnet=False, device=device) 
        
        images = pipe(
            prompt=prompt,
            negative_prompt="(worst quality:2), (low quality:1.8), (normal quality:1.6), bad-hands-5,",
            guidance_scale=7,
            num_inference_steps=40,
            num_images_per_prompt=num_images
        ).images

        return images