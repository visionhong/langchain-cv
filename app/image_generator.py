import streamlit as st
from st_pages import add_page_title
import streamlit.components.v1 as components

import os
from utils.template import image_generate_template
from utils.agent import image_generator_agent
from utils.action import select_image


def image_generator(): 
    add_page_title()
    
    if "generted_image" not in st.session_state:
        st.session_state["generted_image"] = False
        
    num_images = st.slider('Number of images to generate', 1, 8, 2)
    prompt = st.chat_input("Send a message")
    
    if prompt:
        with st.chat_message("assistant"):
            st.write("AI")
            with st.spinner(text="In progress..."):
                agent = image_generator_agent()
                agent(image_generate_template(prompt=prompt, num_images=num_images))
                st.session_state["generted_image"] = True
                
                st.experimental_rerun()
                
    if st.session_state["generted_image"]:
        
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