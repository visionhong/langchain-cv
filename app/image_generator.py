import streamlit as st
from st_pages import add_page_title
from streamlit_drawable_canvas import st_canvas

import streamlit.components.v1 as components

import os
from PIL import Image
import numpy as np


from utils.template import image_generate_template
from utils.agent import image_generator_agent
from utils.action import select_image
from utils.util import get_canny_image, resize_image


def image_generator(): 
    add_page_title()
    
    if "generted_image" not in st.session_state:
        st.session_state["generted_image"] = False
        
    
    mode = st.radio(
    "Choose a guide for generate image",
    ["Nothing", "Image", "Sketch"],
    captions=["","","Generate images based on your sketch."],
    on_change=st.session_state.clear)
        
    num_images = st.slider('Number of images to generate', 1, 8, 2)    
        
    if mode == "Sketch":
        stroke_color = st.sidebar.color_picker("Annotation color: ", "#141412")
        drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
        )

        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)      

        canvas = st_canvas(
            fill_color="#14141277",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            height=704,
            width=704,
            drawing_mode=drawing_mode,
            # background_color="#141412",
            key="canvas",
            update_streamlit=True
        )
        
    elif mode == "Image":
        uploaded_image = st.file_uploader("Upload a your image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert('RGB')
            resized_image, _ = resize_image(image, max_width=704, max_height=704)
            st.markdown("#### Origin Image")
            st.image(resized_image)
            
            st.session_state["canny_image"] = get_canny_image(np.array(resized_image))
            st.session_state["use_controlnet"] = True     
            
    else:
        st.session_state["use_controlnet"] = False
                    
    prompt = st.chat_input("Send a message")
    if prompt:
        if mode == "Sketch":
            st.session_state["sketch_image"] = Image.fromarray(canvas.image_data[:, :, -1])
            st.session_state["use_controlnet"] = True  

        with st.chat_message("assistant"):
            st.write("AI")
            with st.spinner(text="In progress..."):
                agent = image_generator_agent()
                agent(image_generate_template(prompt=prompt, num_images=num_images))
                st.session_state["generted_image"] = True
                
                st.experimental_rerun()
                
    if st.session_state["generted_image"]:
        st.markdown("#### Generated Image")
        imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")
        imageUrls = [f"images/" + i for i in os.listdir("./frontend/public/images")]

        selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

        if selectedImageUrl is not None:
            st.image(selectedImageUrl)
            st.session_state["selectedImage"] = os.path.basename(selectedImageUrl)
            
            # buf = BytesIO()
            # st.session_state["inference_image"][st.session_state["image_state"]].save(buf, format="JPEG")
            # byte_im = buf.getvalue()

            # col4.download_button(
            #     label="Download",
            #     data=byte_im,
            #     file_name=uploaded_image.name,
            #     mime="image/png",
            # )
        
        
                   
    
        

                    
if __name__ == "__main__":
    image_generator()