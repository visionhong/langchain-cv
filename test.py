
import streamlit as st
import os
import torch

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline

from dotenv import load_dotenv

from utils.lama_cleaner_helper import load_jit_model
import tritonclient.http


device='cuda'
            
controlnet = ControlNetModel.from_pretrained("/volume/checkpoints/sketch").to(device) 
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "/volume/checkpoints/realistic",
    torch_dtype=torch.float16,
    controlnet=controlnet,
).to(device)