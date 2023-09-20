import sys
sys.path.append('Grounded-Segment-Anything')
sys.path.append('Grounded-Segment-Anything/EfficientSAM')

import streamlit as st

import os
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image, ImageOps
import supervision as sv

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from diffusers import AutoPipelineForInpainting
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

from groundingdino.util.inference import Model
from EfficientSAM.LightHQSAM.setup_light_hqsam import setup_model
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


from utils.lama_cleaner_helper import norm_img, load_jit_model
from utils.util import combine_masks, random_hex_color


def image_captioner(img_path):
    image = Image.open(img_path).convert('RGB')
    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"  # cuda

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def zero_shot_image_classification(image_path, possible_list):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image = Image.open(image_path)
    inputs = processor(text=possible_list, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    
    return probs

def grounded_sam(image_path, situation_list):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./checkpoints/groundingdino_swint_ogc.pth"
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    
    HQSAM_CHECKPOINT_PATH = "./checkpoints/sam_hq_vit_tiny.pth"
    checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
    
    light_hqsam = setup_model()
    light_hqsam.load_state_dict(checkpoint, strict=True)
    light_hqsam.to(device=DEVICE)

    sam_predictor = SamPredictor(light_hqsam)

    # Predict classes and hyper-param for GroundingDINO
    CLASSES = situation_list
    BOX_THRESHOLD = 0.35
    NMS_THRESHOLD = 0.8

    image = cv2.imread(image_path)
    
    detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=BOX_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]

    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    result_masks = []
    for box in detections.xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    detections.mask = np.array(result_masks)    

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    
    
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)




def instruct_pix2pix(image, prompt):
    model_name = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_name, 
                                                                    torch_dtype=torch.float16, 
                                                                    safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    image = ImageOps.exif_transpose(image)

    images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
    return images

def light_hqsam(image, coordinates):
    image = np.array(image)
        
    HQSAM_CHECKPOINT_PATH = "./checkpoints/sam_hq_vit_tiny.pth"
    checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
    
    light_hqsam = setup_model()
    light_hqsam.load_state_dict(checkpoint, strict=True)
    light_hqsam.to(device="cuda")
    
    sam_predictor = SamPredictor(light_hqsam)
    
    mask, segmented_image = segment(
        sam_predictor=sam_predictor,
        image=image,
        coordinates= coordinates
        )
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)
    
    return image, mask, Image.fromarray(segmented_image)

def sd_inpaint(image: Image.Image, mask: Image.Image, inpaint_prompt):

    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                             torch_dtype=torch.float16, 
                                                             variant="fp16").to("cuda")

    guidance_scale = 8
    num_inference_steps = 20
    strength=0.99

    inpainted_image = pipe(prompt=inpaint_prompt, image=image, mask_image=mask, 
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,strength=strength
                        ,width=mask.size[0], height=mask.size[1]).images[0]
    
    return inpainted_image

def segment(sam_predictor, image, coordinates) -> np.ndarray:
    sam_predictor.set_image(image)
     
    if coordinates.shape[1] == 4: # box
        result_masks = []
        for coord in coordinates:  # box가 여러개인 경우에는 하나씩 처리해야 함
            masks, scores, logits = sam_predictor.predict(
                box=coord,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        
        combined_mask = combine_masks(result_masks) # 여러개의 mask를 or 연산
            
    else: # point
        masks, scores, logits = sam_predictor.predict(
            point_coords=coordinates,
            point_labels=np.ones(len(coordinates)),
            multimask_output=True
        )

        index = np.argmax(scores)
        combined_mask = masks[index][np.newaxis, :, :]
    
    mask_annotator = sv.MaskAnnotator(sv.Color.from_hex(color_hex=random_hex_color()))
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=combined_mask),
        mask=combined_mask
    )
    # detections = detections[detections.area == np.max(detections.area)]
    segmented_image = mask_annotator.annotate(scene=cv2.cvtColor(image, cv2.COLOR_RGB2BGR), detections=detections)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
            
    return combined_mask, segmented_image   
        

def lama_cleaner(image, mask, device):
    LAMA_MODEL_URL = os.environ.get(
        "LAMA_MODEL_URL",
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
    )
    LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")

    model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()
    
    image = norm_img(image)
    mask = norm_img(mask)

    mask = (mask > 0) * 1
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    inpainted_image = model(image, mask)

    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")

    return Image.fromarray(cur_res)

def wurstchen(prompt, num_images, device):
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
    pipeline = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to(device)

    images = pipeline(
        prompt, 
        width=1024,
        height=1024,
        prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
        prior_guidance_scale=4.0,
        num_images_per_prompt=num_images,
    ).images
    
    return images