import sys
sys.path.append('Grounded-Segment-Anything')
sys.path.append('Grounded-Segment-Anything/EfficientSAM')

import os
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image, ImageOps
import supervision as sv

from diffusers import  AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

from groundingdino.util.inference import Model
from segment_anything import SamPredictor

from utils.lama_cleaner_helper import norm_img
from utils.util import combine_masks, random_hex_color
from utils.model_setup import get_sam, get_sd_inpaint, get_lama_cleaner, get_instruct_pix2pix


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
    pipe = get_instruct_pix2pix()
    
    image = ImageOps.exif_transpose(image)
    images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
    return images

def sam(image, coordinates):
    image = np.array(image)
    sam_predictor = get_sam(image)
    
    mask, segmented_image = segment(
        sam_predictor=sam_predictor,
        image=image,
        coordinates= coordinates
        )
    
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)
    
    return image, mask, Image.fromarray(segmented_image)

def segment(sam_predictor, image, coordinates) -> np.ndarray:

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