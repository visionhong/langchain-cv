import streamlit as st


def backward_inference_image():
    if st.session_state["image_state"] > 0:
        st.session_state["image_state"] -= 1
    
def forward_inference_image():
    if len(st.session_state["inference_image"]) - 1 > st.session_state["image_state"]:
        st.session_state["image_state"] += 1

def reset_inference_image():
    st.session_state["inference_image"] = [st.session_state["inference_image"][0]]
    st.session_state["image_state"] =0
    
    st.session_state["req_history"] = []
    st.session_state["res_history"] = []
    
def reset_text():
    st.session_state["text"] = ""