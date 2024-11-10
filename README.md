# Sarcasm and Sentiment Detection App

This project is a Streamlit-based web application that allows users to analyze sentiment, sarcasm, and emotions from various text inputs, including tweets and YouTube comments. It utilizes a Flask backend that processes the data with deep learning models, detecting emotions and sarcasm in real-time. 

## Features

- **Tweet Analysis**: Detects sentiment, sarcasm, and emotions in tweets. Displays the primary emotion and sarcasm prediction along with an emotion distribution chart and word cloud.
- **YouTube Comment Analysis**: Analyzes comments from a given YouTube URL, detecting emotions and sarcasm in individual comments, and visualizing the emotion distribution.
- **Deep Learning Emotion Prediction**: Uses deep learning models to predict emotions in any given text. The emotion probabilities are displayed as a bar plot for better visualization.

## Setup and Installation

### Requirements
1. Python 3.x
2. Flask
3. Streamlit
4. Requests
5. Matplotlib
6. Seaborn
7. Pillow
8. TensorFlow or any required deep learning library for emotion detection

### Installation

Clone the repository:

Navigate to the project directory:


cd sentiment-sarcasm-detection
Create and activate a virtual environment (optional but recommended):

bash
Copy code
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Running the Application
Start the Flask backend server for emotion and sarcasm detection:
bash
Copy code
python app.py
Start the Streamlit frontend:
bash
Copy code
streamlit run streamlit_app.py
The app will open in your default browser, and you can start analyzing tweets, YouTube comments, and text for emotions and sarcasm.
Functionality
Analyze Tweet: Users can input a tweet, and the backend will predict the sentiment, sarcasm, and primary emotion in the tweet. It also displays emotion distribution charts and word clouds if available.

Analyze YouTube Comments: By inputting a YouTube URL, users can analyze the comments for sarcasm and emotional sentiment. The app also visualizes the emotion distribution of the comments.

Deep Learning Emotion Prediction: Users can input any text to receive emotion predictions from a pre-trained deep learning model. The app visualizes the emotion probabilities using a bar plot.
