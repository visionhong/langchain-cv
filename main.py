import streamlit as st
import streamlit_authenticator as stauth
from st_pages import show_pages_from_config, add_page_title

import os
from dotenv import load_dotenv
import yaml
from yaml.loader import SafeLoader

import openai

def main():
    hide_bar= """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            visibility:hidden;
            width: 0px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            visibility:hidden;
        }
        </style>
    """

    with open('.streamlit/credentials.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
        
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )

    name, authentication_status, username = authenticator.login('Login', 'main')
        
    if authentication_status == False:
        st.error("Username/password is incorrect")
        st.markdown(hide_bar, unsafe_allow_html=True)

    elif authentication_status == None:
        st.warning("Please enter your username and password")
        st.markdown(hide_bar, unsafe_allow_html=True)
        
    elif authentication_status:        
        st.toast(f'Welcome *{username}* 😊')
                
        load_dotenv()
        # Load the OpenAI API key from the environment variable
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
            
            st.write("")
            _, col = st.columns((15, 2))
            with col:
                authenticator.logout('Logout', 'main')
        
    

if __name__ == "__main__":
    main()
    
    
