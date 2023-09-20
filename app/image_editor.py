import streamlit as st
from st_pages import add_page_title
from streamlit_drawable_canvas import st_canvas
import supervision as sv
import pandas as pd
import cv2
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np

from utils.util import resize_image, xywh2xyxy
from utils.template import inference_template
from utils.agent import image_editor_agent
from utils.action import backward_inference_image, forward_inference_image, reset_inference_image
from utils.inference import light_hqsam

def image_editor():
    add_page_title()
    
    st.caption("기능: image captioning, zero-shot image classification, zero-shot object detection, image2image, inpaint, erase", unsafe_allow_html=True)
    
    if "req_history" not in st.session_state:
        st.session_state["req_history"] = []
        st.session_state["res_history"] = []


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
            
        # coord를 모두 지웠을 때 맨 처음 sam을 지우기 ㄴ위함
        if "canvas" in st.session_state and st.session_state["canvas"] is not None:  
            df = pd.json_normalize(st.session_state["canvas"]['raw']["objects"])
            if st.session_state["sam_image"] != None and len(df)== 0:
                st.session_state["num_coord"] = 0
                st.experimental_rerun()
        
        
        drawing_mode = st.selectbox("Drawing tool:", ("point", "rect", "freedraw"))
        if drawing_mode == "freedraw":
            anno_color = st.color_picker("Annotation color: ", "#141412") + "77"
        else:
            anno_color = st.color_picker("Annotation color: ", "#EA1010") + "77"
        
        canvas = st_canvas(
            fill_color=anno_color,
            stroke_width=40 if drawing_mode == "freedraw" else 2,
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

        
        chat = st.expander("Chat History", expanded=True)
        with chat:
            for (user, ai) in zip(st.session_state["req_history"], st.session_state["res_history"]):
                with st.chat_message("user"):
                    st.write(user)
                    
                with st.chat_message("assistant"):
                    if isinstance(ai, str):
                        st.markdown(ai)
                    
            if prompt:  # 채팅 엔터를 누르자 마자 사용자의 입력이 바로 보이도록 함
                with st.chat_message("user"):
                    st.write(prompt)

        
        if st.session_state["canvas"] is not None:       
            df = pd.json_normalize(st.session_state["canvas"]['raw']["objects"])
            if len(df) == 0:
                st.session_state["num_coord"] = 0
            if len(df) != 0 and st.session_state["num_coord"] != len(df):
                st.session_state["num_coord"] = len(df)
                st.session_state["mask"] = True
                
                if drawing_mode == "rect":
                    coordinates = xywh2xyxy(df[["left", "top", "width", "height"]].values)
                    
                    image, mask, segmented_image = light_hqsam(st.session_state["inference_image"][st.session_state["image_state"]], coordinates)

                    st.session_state["sam_image"] = segmented_image
                    st.session_state["mask"] = Image.fromarray(mask)
                    st.experimental_rerun()
                    
                elif drawing_mode == "point":
                    coordinates = df[["left", "top"]].values
                    
                    image, mask, segmented_image = light_hqsam(st.session_state["inference_image"][st.session_state["image_state"]], coordinates)
                    
                    st.session_state["sam_image"] = segmented_image
                    st.session_state["mask"] = Image.fromarray(mask)
                    st.experimental_rerun()
                    
                elif drawing_mode == "freedraw":
                    st.session_state["mask"] = canvas.image_data[:, :, -1] > 0
            else:
                st.session_state["mask"] = False
    

            
                
               
            
            
        
        if prompt:
            st.session_state["req_history"].append(prompt)
            with chat:
                with st.chat_message("assistant"):
                    st.write("AI")
                    with st.spinner(text="In progress..."):
                            
                                    
                                    
                        agent = image_editor_agent()                    
                        response=agent(inference_template(prompt=prompt))
        
                        if isinstance(response['intermediate_steps'][-1][1], str):  # Tool의 return 값이 string인 경우에 아래코드 수행 (자연스러운 한국어 번역을 위함)
                            st.session_state["res_history"].append(response['output'])
                            
            st.experimental_rerun()  # chat history 업데이트를 위함
    else:
        st.session_state.clear()               


        
if __name__ == "__main__":
    image_editor()