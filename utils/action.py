import streamlit as st


def backward_inference_image():
    if st.session_state["image_state"] > 0:
        st.session_state["image_state"] -= 1
        st.session_state["num_coord"] = 0
    else:
        st.toast('This is First Image!')
    
def forward_inference_image():
    if len(st.session_state["inference_image"]) - 1 > st.session_state["image_state"]:
        st.session_state["image_state"] += 1
        st.session_state["num_coord"] = 0
        st.session_state["canvas"]['raw']["objects"] = []
    else:
        st.toast('This is Last Image!')
            
def reset_inference_image():
    st.session_state["inference_image"] = [st.session_state["inference_image"][0]]
    st.session_state["image_state"] = 0
    st.session_state["num_coord"] = 0
    
def reset_text():
    st.session_state["text"] = ""
    
def select_image():
    st.session_state["confirm"] = True
    
def reset_coord():
    del st.session_state["canvas"]