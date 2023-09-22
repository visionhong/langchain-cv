import sys
sys.path.append('Grounded-Segment-Anything')
sys.path.append('Grounded-Segment-Anything/EfficientSAM')

import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional, Type, List
import numpy as np
import torch
from PIL import Image
import supervision as sv
from langchain.tools import BaseTool

from utils.inference import (
    image_captioner,
    zero_shot_image_classification,
    grounded_sam,
    instruct_pix2pix,
    sam,
    sd_inpaint,
    lama_cleaner,
    wurstchen
)


def sam_inpaint(image, prompt, coordinates):
    image, mask = sam(image, coordinates)
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    
    inpainted_image = sd_inpaint(image, mask, prompt)
    return inpainted_image


def image_transform(prompt):
    image = st.session_state["inference_image"][st.session_state["image_state"]]
    
    if st.session_state["coord"] == False:  # 마스크가 없다면 이미지 전체 변환(pix2pix)
        transform_pillow = instruct_pix2pix(image, prompt)[0]
    else: # 마스크가 있다면 특정 영역만 inpaint
        mask = Image.fromarray(st.session_state["mask"])
        transform_pillow = sd_inpaint(image, mask, prompt)
            
    
    return transform_pillow

def object_erase(image, mask, device):
    transform_pillow = lama_cleaner(image, mask, device)
    return transform_pillow


class ImageCaptionCheckInput(BaseModel):
    """input_path check that is the input for image captioning."""
    img_path: str = Field(..., description="image path for model input image")
    
class ZeroShotImageClassificationCheckInput(BaseModel):
    """input_path check that is the input for zero-shot image classifier."""
    img_path: str = Field(..., description="image path for model input image")
    possible_situation_list: List[str] = Field(..., description="A list of likely situations for which the model predicts the highest situation probability.")

class ZeroShotObjectDetectoonCheckInput(BaseModel):
    """input_path check that is the input for zero-shot object detector."""
    img_path: str = Field(..., description="image path for model input image")
    class_list: List[str] = Field(..., description="List of situation the prompt is trying to find in image")

class ImageTransformCheckInput(BaseModel):
    prompt: str = Field(..., description="prompt for inpainting the image")


class ImageCaptionTool(BaseTool):
    name = "image-captioner"
    description = "Use this tool when you want to describe or summarize an entire part of an image"
    
    def _run(self, img_path: str):
        caption = image_captioner(img_path)
        return caption


    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    
    
    args_schema: Optional[Type[BaseModel]] = ImageCaptionCheckInput
 
class ZeroShotImageClassificationTool(BaseTool):  # TODO: canvas에서 박스를 쳤다면 그 영역 안에서 분류하는 기능 추가 
    name = "zero_shot_image_classification"
    description = "Please use this tool when you need to inquire about the existence of certain objects or when you need to determine which of A and B is correct."
    
    def _run(self, img_path: str, possible_situation_list: List[str]):
        probs = zero_shot_image_classification(img_path, possible_situation_list)
        
        highest_prob_idx = probs.argmax().item()
        
        return f"{probs[0][highest_prob_idx]*100:.1f}% 확률로 {possible_situation_list[highest_prob_idx]} 입니다."
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = ZeroShotImageClassificationCheckInput

class ZeroShotObjectDetectoonTool(BaseTool):  # TODO: 분류하고 detection 구분
    name = "grounded_sam"             
    description = "Please use this tool when you want to locate or determine the presence of a specific object."
    
    def _run(self, img_path: str, class_list: List[str]):
        annotated_image = grounded_sam(img_path, class_list)
        annotated_pillow = Image.fromarray(annotated_image)
        st.session_state["ai_history"].append(annotated_pillow)
        
        torch.cuda.empty_cache()
        return annotated_pillow
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = ZeroShotObjectDetectoonCheckInput

class ImageTransformTool(BaseTool):
    name = "image_transform"
    description = """
    Please use this tool when you want to change the image style or replace, add specific objects with something else.
    """
    
    def _run(self, prompt: str):
        transform_pillow = image_transform(prompt)
        st.session_state["inference_image"].insert(st.session_state["image_state"]+1, transform_pillow)
        st.session_state["image_state"] += 1
        
        torch.cuda.empty_cache()
        return True
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = ImageTransformCheckInput

class ObjectEraseTool(BaseTool):  
    name = "object_erase"
    description = """
    Please use this tool when you want to clean, erase or delete certain objects from an image.
    """
    
    def _run(self, prompt=None):
        pil_image = st.session_state["inference_image"][st.session_state["image_state"]]
        np_image = np.array(pil_image)
        try:
            mask = st.session_state["mask"]
        except:
            st.exception(RuntimeError('There is no mask. Please masking the object in the drawing tool.'))
            return False

        transform_pillow = object_erase(np_image, mask, "cuda")
    
        st.session_state["inference_image"].insert(st.session_state["image_state"]+1, transform_pillow)
        st.session_state["image_state"] += 1
        
        torch.cuda.empty_cache()
        return True
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")



class ImageGenerationCheckInput(BaseModel):
    prompt: str = Field(..., description="prompt for generating image")
    num_images: int = Field(..., description="number of images to generate")

class ImageGenerationTool(BaseTool):  
    name = "wurstchen"
    description = """
    Please use this tool when you want to generate images.
    """
    
    def _run(self, prompt: str, num_images: int, device: str = "cuda"):
        
        images = wurstchen(prompt, num_images, device) 
        
        for idx, image in enumerate(images):
            image.save(f"./frontend/public/images/{idx}.png")
        
        
        torch.cuda.empty_cache()
        return "complete!"
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = ImageGenerationCheckInput
