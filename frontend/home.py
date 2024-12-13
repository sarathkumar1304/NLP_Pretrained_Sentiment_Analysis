import streamlit as st


def home_ui():

    # Streamlit app title
    st.title("Transformer-Based Text Classification Project 🌟")

    # Objective section
    st.header("🔍 Objective")
    st.write("""
    The primary objective of this project is to classify text into positive or negative sentiment using a 
    **Transformer-based pre-trained model**. This model helps in understanding the sentiment of user-provided text, 
    which can be useful in applications like customer feedback analysis, review classification, and more.
    """)

    # Tools used
    st.header("🛠️ Tools Used")
    st.write("""
    This project leverages the following tools and technologies:
    - **Python**: For data preprocessing and backend logic.
    - **Hugging Face Transformers**: For leveraging pre-trained Transformer models.
    - **PyTorch**: For model operations and predictions.
    - **Docker** (optional): To containerize the application for deployment.
    - **Mlflow** : For model tracking and version control.
    - **Git**: For version control and collaboration.
    - **Streamlit**: To create an interactive and user-friendly UI.
    """)

    # Architecture section
    st.header("🏗️ Project Architecture")
    st.write("""
    The architecture of this project can be summarized in the following flow:
    """)

    # Display architecture image
    # architecture_image_path = "path_to_your_architecture_image.png"  # Replace with your image path
    # st.image(architecture_image_path, caption="Project Architecture", use_column_width=True)

    # Footer or additional information
    st.write("---")
    st.write("""
    💡 This application is designed to showcase the integration of **NLP** and **Machine Learning** with 
    an easy-to-use web interface. The predictions are generated in real-time, providing insights into text sentiments.
    """)
