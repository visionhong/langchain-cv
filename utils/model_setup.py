
import streamlit as st
import os
import torch

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline

from dotenv import load_dotenv

from utils.lama_cleaner_helper import load_jit_model
import tritonclient.http


def get_triton_client():
    load_dotenv()
    url = os.getenv("TRITON_HTTP_URL")
    
    triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)  
      
    return triton_client


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


@st.cache_resource
def get_general_generator(use_controlnet=False, device='cuda'):
    if use_controlnet:
        if "sketch_image" in st.session_state and st.session_state["sketch_image"] != None:
            controlnet = ControlNetModel.from_pretrained("/volume/checkpoints/sketch").to(device)
            model_name = "sketch_image"
            print("use sketch net")
        else:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device)
            model_name = "canny_image"
         
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "/volume/checkpoints/realistic",
            torch_dtype=torch.float16,
            controlnet=controlnet,
        ).to(device)
        
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "/volume/checkpoints/realistic",
            torch_dtype=torch.float16,
        ).to(device)
        
        model_name=None
        
    return pipeline, model_name


@st.cache_resource
def get_male_anime_generator(use_controlnet=False, device='cuda'):
    if use_controlnet:
        if "sketch_image" in st.session_state and st.session_state["sketch_image"] != None:
            controlnet = ControlNetModel.from_pretrained("/volume/checkpoints/sketch").to(device)
            model_name = "sketch_image"
            print("use sketch net")
        else:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device)
            model_name = "canny_image"
            
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "/volume/checkpoints/somman",  # This line
            safety_checker=None,
            controlnet=controlnet,
        ).to(device)
        
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "/volume/checkpoints/somman",  # This line
            safety_checker=None,
        ).to(device)
        model_name=None
        
    return pipeline, model_name


@st.cache_resource
def get_female_anime_generator(use_controlnet=False, device='cuda'):
    if use_controlnet:
        if "sketch_image" in st.session_state and st.session_state["sketch_image"] != None:
            controlnet = ControlNetModel.from_pretrained("/volume/checkpoints/sketch").to(device)
            model_name = "sketch_image"
            print("use sketch net")
        else:
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device)
            model_name = "canny_image"
            
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "/volume/checkpoints/female",  # This line
            safety_checker=None,
            controlnet=controlnet,
        ).to(device)
        
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "/volume/checkpoints/female",  # This line
            safety_checker=None,
        ).to(device)
        model_name=None
        
    return pipeline, model_name