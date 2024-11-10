import pickle
import requests
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from wordcloud import WordCloud
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from googleapiclient.discovery import build
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize the Flask app
app = Flask(__name__)

# Load the individual components (model, vectorizer, label encoder) from the .pkl file
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model_data = pickle.load(f)
            print("Model, TF-IDF Vectorizer, and Label Encoder loaded successfully.")
            return model_data['model'], model_data['tfidf_vectorizer'], model_data['label_encoder']
    except Exception as e:
        print(f"Error loading model components: {e}")
        return None, None, None  # Handle failed load

rf, tf, label_encoder = load_model()

# Set up the YouTube API
YOUTUBE_API_KEY = "AIzaSyAXjgC8zQSrma6jenWEDNlzySY0L884NqE"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Dummy sarcasm detection function (implement your own logic)
def detect_sarcasm(text):
    sarcastic_keywords = ["love", "fantastic", "great", "amazing", "wonderful"]
    if any(keyword in text.lower() for keyword in sarcastic_keywords) and ("not" in text.lower() or "!" in text):
        return True
    return False

# Preprocessing function (assuming you have some preprocessing steps like lowercasing, stopword removal, etc.)
def preprocess_text(text):
    return text.lower()

# Emotion prediction function using RandomForestClassifier, TfidfVectorizer, and LabelEncoder
def predict_emotion(text, rf, tf, label_encoder):
    if rf is None or tf is None or label_encoder is None:
        raise ValueError("Model components are not properly loaded.")
    
    try:
        # Preprocess and transform the input text using the fitted TF-IDF vectorizer
        text_cleaned = preprocess_text(text)  # Apply preprocessing
        text_transformed = tf.transform([text_cleaned])  # Use tf.transform to convert text into numerical features

        # Predict the emotion using the trained model (RandomForest)
        predicted_label = rf.predict(text_transformed)

        # Decode the numeric label back to the original emotion label using LabelEncoder
        emotion = label_encoder.inverse_transform(predicted_label)
        return emotion[0]  # Return the emotion name
    except Exception as e:
        print(f"Error in predict_emotion function: {e}")
        return None  # Return None in case of an error

# Function to create emotion bar chart
def create_emotion_chart(probs):
    try:
        emotion_labels = ["happy", "sad", "angry", "disgust", "fear", "surprise"]
        fig, ax = plt.subplots()
        ax.bar(emotion_labels, probs, color=['yellow', 'blue', 'red', 'purple', 'gray', 'orange'])
        ax.set_title("Emotion Distribution")
        ax.set_ylim(0, 1)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64
    except Exception as e:
        print(f"Error creating emotion chart: {e}")
        return None

# Function to create a word cloud for the primary emotion
def create_word_cloud(emotion):
    try:
        words = f"{emotion} words " * 10
        wordcloud = WordCloud(width=400, height=400).generate(words)
        buf = io.BytesIO()
        wordcloud.to_image().save(buf, format="PNG")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64
    except Exception as e:
        print(f"Error creating word cloud: {e}")
        return None

# Function to fetch comments from a YouTube video using YouTube API
def fetch_youtube_comments(video_url):
    # Improved video ID extraction with regex
    video_id = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", video_url)
    if not video_id:
        return []
    video_id = video_id.group(1)  # Extracted video ID
    comments = []
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText"
        )
        response = request.execute()
        
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        
        return comments
    except Exception as e:
        print(f"Error fetching YouTube comments: {e}")
        return []

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion_endpoint():
    data = request.get_json()
    tweet = data.get("tweet", "")
    
    if not tweet:
        return jsonify({"error": "No tweet provided"}), 400

    # Predict the emotion using the model (RandomForest, TF-IDF, LabelEncoder)
    try:
        predicted_emotion = predict_emotion(tweet, rf, tf, label_encoder)
        if predicted_emotion is None:
            return jsonify({"error": "Error predicting emotion"}), 500
    except Exception as e:
        return jsonify({"error": f"Error in emotion prediction: {str(e)}"}), 500

    # Assign probabilities for the emotions (placeholder)
    emotion_labels = ["happy", "sad", "angry", "disgust", "fear", "surprise"]
    emotion_probs = [0] * len(emotion_labels)  # Placeholder probabilities
    emotion_probs[emotion_labels.index(predicted_emotion)] = 1.0  # Set the predicted emotion probability to 1

    # Check if sarcasm is detected
    sarcasm_detected = detect_sarcasm(tweet)
    sarcasm_prediction = "Likely sarcastic" if sarcasm_detected else "Not sarcastic or can't determine"

    # Generate visual elements
    emotion_chart = create_emotion_chart(emotion_probs)
    emotion_wordcloud = create_word_cloud(predicted_emotion)

    return jsonify({
        "primary_emotion": predicted_emotion,
        "sarcasm_prediction": sarcasm_prediction,
        "emotion_chart": emotion_chart,
        "emotion_wordcloud": emotion_wordcloud,
        "ai_response": "Analyzed successfully",
        "emotion_icon": "😃" if predicted_emotion == "happy" else "😟"  # Adjust icons based on emotion
    })

@app.route("/analyze_youtube_comments", methods=["POST"])
def analyze_youtube_comments():
    data = request.get_json()
    youtube_url = data.get("youtube_url", "")
    
    if not youtube_url:
        return jsonify({"error": "No YouTube URL provided"}), 400
    
    # Fetch comments from YouTube
    comments = fetch_youtube_comments(youtube_url)
    if not comments:
        return jsonify({"error": "Error fetching comments from YouTube"}), 500

    # Analyze each comment for emotion and sarcasm
    analyzed_comments = []
    for comment in comments:
        predicted_emotion = predict_emotion(comment, rf, tf, label_encoder)
        sarcasm_detected = detect_sarcasm(comment)
        sarcasm_prediction = "Likely sarcastic" if sarcasm_detected else "Not sarcastic or can't determine"
        
        analyzed_comments.append({
            "comment": comment,
            "predicted_emotion": predicted_emotion,
            "sarcasm_prediction": sarcasm_prediction
        })
    
    return jsonify({
        "comments": analyzed_comments,
        "ai_response": "Comments analyzed successfully"
    })

# Deep learning model for emotion classification using Hugging Face
tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

# Route for deep learning-based emotion prediction
@app.route('/deep_learning_predict', methods=['POST'])
def deep_learning_predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Apply sigmoid and get probabilities
    probs = torch.sigmoid(logits).squeeze().detach().numpy()

    # Define emotion labels
    labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
        "confusion", "curiosity", "desire", "disappointment", "disapproval", 
        "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", 
        "love", "nervousness", "optimism", "pride", "realization", "relief", 
        "remorse", "sadness", "surprise"
    ]

    # Convert float32 to float for JSON serialization
    emotion_probs = {label: float(prob) for label, prob in zip(labels, probs)}

    return jsonify(emotion_probs)


if __name__ == "__main__":
    app.run(debug=True)
