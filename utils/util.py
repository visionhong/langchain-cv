import os
import cv2
from PIL import Image
import re
import numpy as np
import random


def save_uploaded_image(directory, uploaded_images) :
    os.makedirs(directory, exist_ok=True)
    
    for image in uploaded_images:
        with open(os.path.join(directory, image.name), 'wb') as f :
            f.write(image.getbuffer())
            
            
def save_uploaded_file(directory, file):
    os.makedirs(directory, exist_ok=True)
    
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())


def save_dataframe(directory, file_name, df):
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, file_name), index=False)

        
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)


def closest_multiple_of_8(number):
    remainder = number % 8
    
    if remainder < 4:
        result = number - remainder
    else:
        result = number + (8 - remainder)
    
    return result
   
   
def resize_image(image, max_width, max_height):
    width, height = image.size
    
    if width > max_width or height > max_height:
        width_ratio = max_width / width
        height_ratio = max_height / height
        
        resize_ratio = min(width_ratio, height_ratio)
        new_width = closest_multiple_of_8(int(width * resize_ratio))
        new_height = closest_multiple_of_8(int(height * resize_ratio))        
        
        resized_img = image.resize((new_width, new_height), Image.ANTIALIAS)
    else:
        resized_img = image
        resize_ratio = 0
        
    return resized_img, resize_ratio
  
            
def plot_bboxes(image, box, score, label, label_index):

    # colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
    colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]

    label = label + " " + str(score) + "%"

    color = colors[label_index]
    box_label(image, box, label, color)
    
    
def string_to_dictionary(text):
  sections = text.split('\n\n')
  result_dict = {}

  for section in sections:
      parts = section.split(':')
      
      key = parts[0].strip()
      value = parts[1].strip()
      
      result_dict[key] = value

  return result_dict


def label_select(text):
    result = re.search(r'start:(.*?):end', text)
    extracted_text = result.group(1)
    return extracted_text
    
def xywh2xyxy(xywh):
    xyxy = xywh.copy()

    xyxy[:,2] = xywh[:, 0] + xywh[:, 2]
    xyxy[:,3] = xywh[:, 1] + xywh[:, 3]
    return xyxy


def combine_masks(mask_list):
    if not mask_list:
        raise ValueError("적어도 하나 이상의 배열이 필요합니다.")
    
    combined_mask = mask_list[0]  # 첫 번째 배열을 초기값으로 설정합니다.
    
    # 나머지 배열들을 OR 연산하여 결합합니다.
    for mask in mask_list[1:]:
        combined_mask = np.logical_or(combined_mask, mask)
    
    return combined_mask[np.newaxis, :, :]

def random_hex_color():
    color_list = ["#caff70", "#07ccff", "#fa0087", "#f88379", "#7d37ff"]
    hex_color = random.choice(color_list)
    
    return hex_color


def dilate_mask(mask, kernel_size=3, iterations=1):
    mask = mask.astype(np.uint8).squeeze()
    # 커널 생성 (모든 방향으로 팽창)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 팽창 연산 적용
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    return dilated_mask


def get_canny_image(image: np.ndarray):
    low_threshold = 100
    high_threshold = 200
    # do canny edge detection
    image_canny = cv2.Canny(image, low_threshold, high_threshold)
    # convert to PIL image format
    image_canny = image_canny[:, :, None]
    image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
    image_canny = Image.fromarray(image_canny)
    return image_canny