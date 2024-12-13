import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from about import about_me_ui
from project import project_ui
from home import home_ui


with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Project","About Me"],
        icons=["house", "app-indicator" ,"person-video3"],
        menu_icon="cast",
        default_index=1,
    )

if selected == "Home":
    home_ui()

if selected == "Project":
    project_ui()

if selected == "About Me":
    about_me_ui()