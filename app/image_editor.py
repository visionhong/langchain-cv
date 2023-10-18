import streamlit as st
from st_pages import add_page_title
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np

from utils.util import resize_image, xywh2xyxy
from utils.template import image_editor_template
from utils.agent import image_editor_agent
from utils.action import backward_inference_image, forward_inference_image, reset_inference_image, reset_coord
from utils.inference import sam

def image_editor():
    add_page_title()
    
    st.caption("기능: zero-shot segmentation, image2image, inpaint, erase", unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload a your image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:  
        prompt = st.chat_input("Send a message")
        
        if "image_state" not in st.session_state:
            image = Image.open(uploaded_image).convert('RGB')
            image = ImageOps.exif_transpose(image)  # 이미지 자동회전 금지
            resized_image, resize_ratio = resize_image(image, max_width=704, max_height=704)
            
            st.session_state["image_state"] = 0
            st.session_state["inference_image"] = [resized_image]
            st.session_state["sam_image"] = None
            st.session_state["num_coord"] = 0
            
        # coord를 모두 지웠을 때 맨 처음 sam을 지우기 위함
        if "canvas" in st.session_state and st.session_state["canvas"] is not None:  
            df = pd.json_normalize(st.session_state["canvas"]['raw']["objects"])
            if st.session_state["sam_image"] != None and len(df)== 0:
                st.session_state["num_coord"] = 0
                st.session_state["sam_image"] = None
                # st.experimental_rerun()
        
        
        drawing_mode = st.selectbox("Drawing tool:", ("rect", "point", "freedraw"), on_change=reset_coord)
        if drawing_mode == "freedraw":
            col1, col2 = st.columns(2)
            anno_color = col1.color_picker("Annotation color: ", "#141412") + "77"
            brush_width = col2.number_input("Brush width", value=40)
        else:
            anno_color = st.color_picker("Annotation color: ", "#141412") + "77"
        
        col1, col2 = st.columns((0.1,1))
        with col2:   
            canvas = st_canvas(
                fill_color=anno_color,
                stroke_width=brush_width if drawing_mode == "freedraw" else 2,
                stroke_color="black" if drawing_mode != "freedraw" else anno_color,
                background_image=st.session_state["sam_image"] if st.session_state["num_coord"] != 0 else st.session_state["inference_image"][st.session_state["image_state"]],
                height=st.session_state["inference_image"][0].height,
                width=st.session_state["inference_image"][0].width,
                drawing_mode=drawing_mode,
                key="canvas",
                point_display_radius=4,
                update_streamlit=True
            )
            
               
        col1, col2, _, col3, col4 = st.columns((4,4,10,3,4))
        col1.button("backward", on_click=backward_inference_image, use_container_width=True)
        col2.button("forward", on_click=forward_inference_image, use_container_width=True)
        col3.button("reset", on_click=reset_inference_image, use_container_width=True)
        
        buf = BytesIO()
        st.session_state["inference_image"][st.session_state["image_state"]].save(buf, format="JPEG")
        byte_im = buf.getvalue()

        col4.download_button(
            label="Download",
            data=byte_im,
            file_name=uploaded_image.name,
            mime="image/png",
        )
        
        
        if st.session_state["canvas"] is not None:       
            df = pd.json_normalize(st.session_state["canvas"]['raw']["objects"])
            if len(df) == 0:
                st.session_state["coord"] = False
            else:    
                st.session_state["coord"] = True

            
            if len(df) != 0 and st.session_state["num_coord"] != len(df):
                st.session_state["num_coord"] = len(df)
                st.session_state["freedraw"] = False
                
                if drawing_mode == "rect":
                    pos_coords = xywh2xyxy(df[["left", "top", "width", "height"]].values)
                    neg_coords = np.zeros([1, 2])
                    labels = labels = np.array([2, 3])  # box width, box height

                    image, mask, segmented_image = sam(image=st.session_state["inference_image"][st.session_state["image_state"]],
                                                       pos_coords=pos_coords,
                                                       neg_coords=neg_coords,
                                                       labels=labels)
                    
                    st.session_state["sam_image"] = segmented_image
                    st.session_state["mask"] = mask
                    st.experimental_rerun()
                    
                elif drawing_mode == "point":

                    pos_coords = df[["left", "top"]].values
                    neg_coords = np.zeros([1, 2])
                    labels = np.array([1])  # point
                    
                    image, mask, segmented_image = sam(image=st.session_state["inference_image"][st.session_state["image_state"]],
                                                       pos_coords=pos_coords,
                                                       neg_coords=neg_coords,
                                                       labels=labels)
                    
                    
                    st.session_state["sam_image"] = segmented_image
                    st.session_state["mask"] = mask
                    st.experimental_rerun()
                    
                elif drawing_mode == "freedraw":
                    st.session_state["mask"] = canvas.image_data[:, :, -1] > 0
                    st.session_state["freedraw"] = True
                
                    
        if prompt:
            with st.spinner(text="Please wait..."):
                agent = image_editor_agent()                  
                transform_pillow = agent(image_editor_template(prompt=prompt))['output']
                
                st.session_state["inference_image"].insert(st.session_state["image_state"]+1, transform_pillow)
                st.session_state["image_state"] += 1
                
                st.session_state["num_coord"] = 0
                st.session_state["canvas"]['raw']["objects"] = []
                st.experimental_rerun()
    else:
        st.session_state.clear()               


        
if __name__ == "__main__":
    image_editor()