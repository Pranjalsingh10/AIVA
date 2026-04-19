import speech_recognition as sr
import numpy as np
import tensorflow as tf
import pickle
import re

# Load emotion model
model = tf.keras.models.load_model("model/emotion_glove_model.keras")

# Load tokenizer + label encoder (IMPORTANT)
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

max_len = 100

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# Predict emotion
def predict_emotion(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)

    pred = model.predict(pad, verbose=0)
    confidence = np.max(pred)
    emotion = le.inverse_transform([np.argmax(pred)])[0]

    if confidence < 0.5:
        return "neutral"

    return f"{emotion} ({confidence:.2f})"

# Speech to text
def listen_and_predict():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)

        emotion = predict_emotion(text)
        return emotion

    except sr.UnknownValueError:
        return "Could not understand"

    except sr.RequestError:
        return "API error"