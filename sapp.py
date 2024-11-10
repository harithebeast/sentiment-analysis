import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')   # Set the backend to Agg to avoid GUI-related issues

# Set up Streamlit page
st.title("Sarcasm and Sentiment Detection")
st.write("Select an option to analyze sentiment, sarcasm, and emotion.")

# Create a navigation bar with options
page = st.radio("Choose an analysis", ["Analyze Tweet", "Analyze YouTube Comments", "Deep Learning Emotion Prediction"])

# Define the Flask API endpoints for analysis
FLASK_API_URL = "http://127.0.0.1:5000/detect_emotion"
YOUTUBE_API_URL = "http://127.0.0.1:5000/analyze_youtube_comments"
DEEP_LEARNING_API_URL = "http://127.0.0.1:5000/deep_learning_predict"

# ---------------------------- Tweet Analysis Section ----------------------------
if page == "Analyze Tweet":
    st.subheader("Enter a tweet to detect sentiment, sarcasm, and emotion.")
    
    # Input field for the tweet
    tweet = st.text_area("Tweet", "")
    
    # When the button is clicked, send request to Flask backend
    if st.button("Analyze Tweet"):
        if tweet:
            try:
                # Send POST request to Flask API with the tweet data
                response = requests.post(FLASK_API_URL, json={"tweet": tweet})
                
                # Check if the response is successful
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response data
                    primary_emotion = result.get("primary_emotion", "N/A")
                    sarcasm_prediction = result.get("sarcasm_prediction", "N/A")
                    emotion_icon = result.get("emotion_icon", "")
                    emotion_chart = result.get("emotion_chart", "")
                    emotion_wordcloud = result.get("emotion_wordcloud", "")
                    ai_response = result.get("ai_response", "N/A")
                    
                    # Display AI Response
                    st.write(f"**AI Response:** {ai_response}")
                    
                    # Display Emotion, Sarcasm Prediction, and Emotion Icon
                    st.write(f"**Primary Emotion:** {primary_emotion} {emotion_icon}")
                    st.write(f"**Sarcasm Prediction:** {sarcasm_prediction}")
                    
                    # Display Emotion Chart if available
                    if emotion_chart:
                        st.write("**Emotion Distribution:**")
                        emotion_chart_img = base64.b64decode(emotion_chart)
                        img = Image.open(BytesIO(emotion_chart_img))
                        st.image(img, use_column_width=True)
                    
                    # Display Emotion Word Cloud if available
                    if emotion_wordcloud:
                        st.write(f"**Emotion Word Cloud ({primary_emotion}):**")
                        wordcloud_img = base64.b64decode(emotion_wordcloud)
                        img = Image.open(BytesIO(wordcloud_img))
                        st.image(img, use_column_width=True)
                
                else:
                    st.error("Error: Could not analyze the tweet. Please try again.")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Request to Flask API failed: {e}")
        else:
            st.warning("Please enter a tweet to analyze.")

# ---------------------------- YouTube Comment Analysis Section ----------------------------
elif page == "Analyze YouTube Comments":
    st.subheader("Enter a YouTube URL to analyze comments.")
    
    # Input field for the YouTube URL
    youtube_url = st.text_input("YouTube URL", "")
    
    # When the button is clicked, send request to Flask backend
    if st.button("Analyze YouTube Comments"):
        if youtube_url:
            try:
                # Send POST request to Flask API with the YouTube URL
                response = requests.post(YOUTUBE_API_URL, json={"youtube_url": youtube_url})
                
                # Check if the response is successful
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response data (for example, list of comments)
                    comments = result.get("comments", [])
                    emotion_counts = result.get("emotion_counts", {})  # Get emotion distribution from response
                    ai_response = result.get("ai_response", "N/A")
                    
                    # Display AI Response
                    st.write(f"**AI Response:** {ai_response}")
                    
                    # Display Comments
                    st.write(f"**Comments ({len(comments)}):**")
                    for idx, comment in enumerate(comments, 1):
                        st.write(f"{idx}. {comment['comment']}")
                        st.write(f"Predicted Emotion: {comment['predicted_emotion']}, Sarcasm: {comment['sarcasm_prediction']}")
                    
                    # Display the Emotion Distribution Chart
                    if emotion_counts:
                        st.write("**Emotion Distribution of Comments:**")
                        
                        # Plotting the emotion distribution
                        emotion_labels = list(emotion_counts.keys())
                        emotion_values = list(emotion_counts.values())
                        
                        # Use Seaborn or Matplotlib for better visualization
                        fig, ax = plt.subplots()
                        sns.barplot(x=emotion_labels, y=emotion_values, ax=ax, palette='viridis')
                        ax.set_xlabel('Emotion')
                        ax.set_ylabel('Number of Comments')
                        ax.set_title('Emotion Distribution')
                        st.pyplot(fig)
                
                else:
                    st.error("Error: Could not analyze YouTube comments. Please try again.")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Request to Flask API failed: {e}")
        else:
            st.warning("Please enter a YouTube URL to analyze.")

# ---------------------------- Deep Learning Emotion Prediction Section ----------------------------
elif page == "Deep Learning Emotion Prediction":
    st.subheader("Enter text for deep learning-based emotion prediction.")
    
    # Input field for text
    text = st.text_area("Text", "")
    
    # When the button is clicked, send request to Flask backend
    if st.button("Predict Emotion"):
        if text:
            try:
                # Send POST request to Flask API with the text data
                response = requests.post(DEEP_LEARNING_API_URL, json={"text": text})
                
                # Check if the response is successful
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response data (emotion probabilities)
                    if result:
                        st.write("**Emotion Probabilities:**")
                        
                        # Plot the emotion probabilities
                        emotions = list(result.keys())
                        probabilities = list(result.values())
                        
                        # Create a bar plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(emotions, probabilities, color='skyblue')
                        ax.set_xlabel('Probability')
                        ax.set_title('Emotion Probabilities')
                        ax.invert_yaxis()  # Optional: highest probability at the top
                        
                        # Display the plot in Streamlit
                        st.pyplot(fig)
                else:
                    st.error("Error: Could not predict emotion. Please try again.")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Request to Flask API failed: {e}")
        else:
            st.warning("Please enter text to predict emotion.")