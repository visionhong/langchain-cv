
import streamlit as st
import os
import torch

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, AutoPipelineForText2Image, StableDiffusionInpaintPipeline
from segment_anything import SamPredictor

from EfficientSAM.MobileSAM.setup_light_hqsam import setup_model
from utils.lama_cleaner_helper import load_jit_model

@st.cache_resource
def get_sam(image):
    print("Mobile SAM setup!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # checkpoint = torch.load("./checkpoints/sam_hq_vit_tiny.pth")
    # light_hqsam = setup_model()
    # light_hqsam.load_state_dict(checkpoint, strict=True)
    # light_hqsam.to(device)

    checkpoint = torch.load("./checkpoints/mobile_sam.pt")
    mobile_sam = setup_model()
    mobile_sam.load_state_dict(checkpoint, strict=True)
    mobile_sam.to(device)
    
    sam_predictor = SamPredictor(mobile_sam)
    sam_predictor.set_image(image)
    
    return sam_predictor

@st.cache_resource
def get_sd_inpaint():
    print("Stable Diffusion Inpaint setup!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to(device)
    
    return pipe



@st.cache_resource
def get_lama_cleaner():
    print("lama cleaner setup!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    LAMA_MODEL_URL = os.environ.get(
        "LAMA_MODEL_URL",
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
    )
    LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")

    lama_model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()
    return lama_model

@st.cache_resource
def get_instruct_pix2pix():
    print("Instruct Pix2Pix setup!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_name, 
                                                                    torch_dtype=torch.float16, 
                                                                    safety_checker=None).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe