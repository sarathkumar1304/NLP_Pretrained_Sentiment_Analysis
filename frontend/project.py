import streamlit as st
from transformers import pipeline
from streamlit_echarts import st_echarts

def project_ui():
    # Load the pre-trained sentiment analysis model
    model_name = "model"
    classifier = pipeline("sentiment-analysis", model=model_name)

    # App title and description
    st.title("Transformer-Based Text Classification")
    st.write("""
    This app uses a pre-trained Transformer model to classify text. Enter your text below to get the classification result.
    """)

    # User input
    user_input = st.text_area("Enter your text here", height=150)

    # Prediction button
    if st.button("Predict"):
        if user_input.strip():
            try:
                # Perform text classification
                predictions = classifier(user_input)

                # Extract label and score
                label = predictions[0]['label']
                score = predictions[0]['score']

                # Calculate positive and negative scores
                if label == 'LABEL_0':
                    negative_score = score
                    positive_score = 1 - score
                else:
                    positive_score = score
                    negative_score = 1 - score

                # Display sentiment prediction and scores
                if label == 'LABEL_0':
                    st.error("Prediction: 😔 Negative")
                else:
                    st.success("Prediction: 😊 Positive")

                st.write("### Sentiment Scores")
                st.write(f"Positive Score: {positive_score * 100:.2f}%")
                st.write(f"Negative Score: {negative_score * 100:.2f}%")

                # Display interactive sentiment analysis indicator
                options = {
                    "series": [
                        {
                            "type": "gauge",
                            "startAngle": 180,
                            "endAngle": 0,
                            "radius": "100%",
                            "pointer": {"show": True, "length": "60%", "width": 5},
                            "progress": {
                                "show": True,
                                "overlap": False,
                                "roundCap": True,
                                "clip": False
                            },
                            "axisLine": {
                                "lineStyle": {
                                    "width": 10,
                                    "color": [
                                        [0.5, "#FF6F61"],  # Negative (Red)
                                        [1, "#6AA84F"]   # Positive (Green)
                                    ]
                                }
                            },
                            "axisTick": {"show": False},
                            "splitLine": {"show": False},
                            "axisLabel": {"distance": 15, "fontSize": 10},
                            "data": [
                                {"value": positive_score * 100, "name": "Positive"},
                            ],
                            "title": {"fontSize": 14},
                            "detail": {
                                "valueAnimation": True,
                                "formatter": "{value}%",
                                "fontSize": 12
                            },
                            "animation": True,  # Enable animation
                            "animationDuration": 2000,  # Duration in ms
                            "animationEasing": "cubicOut", # Easing function
                    
                        }
                    ]
                }

                st.write("### Sentiment Analysis Indicator")
                st_echarts(options, height="300px")

                # Warning if confidence is below 60%
                if score < 0.6:
                    st.warning("The confidence level of the prediction is below 60%. The result may not be reliable.")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some text for prediction.")

# Run the Streamlit app
if __name__ == "__main__":
    project_ui()

