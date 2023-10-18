import streamlit as st
import streamlit_authenticator as stauth
from st_pages import show_pages_from_config, add_page_title

import os
from dotenv import load_dotenv
import yaml
from yaml.loader import SafeLoader

import openai

def main():
            
    load_dotenv()
    
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        print("OPENAI_API_KEY is set")


    show_pages_from_config(".streamlit/pages.toml")
    add_page_title()
    
    st.write("")
    tab1, tab2 = st.tabs(["Image Editor", "Image Generator"])
    
    with tab1:
        st.markdown(
            """
            ##### Image Editor
            Drawing Tool과 채팅을 통해 이미지에서 객체를 바꾸거나 지우고, 이미지 스타일을 변환할 수 있습니다.
            """
        )

        st.markdown("#### Demo")
        st.video("https://www.youtube.com/watch?v=NcU5xeHIGUE") 
        

if __name__ == "__main__":
    main()
    
    
