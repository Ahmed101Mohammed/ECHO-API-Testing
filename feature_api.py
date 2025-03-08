from fastapi import FastAPI, UploadFile, File
import cv2 as cv
import numpy as np
import mediapipe as mp
import pickle
from gtts import gTTS
import os
from fastapi.responses import JSONResponse, FileResponse

# Load Trained Model
with open('model.pickle', 'rb') as saved_model:
    loaded_model = pickle.load(saved_model)
    model = loaded_model['model']

# Mediapipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Labels
labels = {
    0: 'السلام عليكم',
    1: 'كيف الحال',
    2: 'حذف اخر اشارة'
}

# Store Predicted Labels for Session
predicted_texts = []

# Initialize API
app = FastAPI()

@app.post("/predict/")
async def predict_sign(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        
        # Process Image
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5) as hands:
            results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract Hand Landmark Features
                data_corr = [coord for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y)]

                if len(data_corr) == 42:
                    predicted_class = int(model.predict([np.asarray(data_corr)])[0])
                    predicted_accuracy = model.predict_proba([np.asarray(data_corr)])[0][predicted_class]

                    if predicted_class in labels and predicted_accuracy > 0.5:
                        recognized_text = labels[predicted_class]
                        predicted_texts.append(recognized_text)  # Store for final speech
                        return JSONResponse(content={"sign": recognized_text, "confidence": round(predicted_accuracy * 100, 2)})

        return JSONResponse(content={"sign": "No hand detected", "confidence": 0})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/generate-voice/")
async def generate_voice():
    try:
        if not predicted_texts:
            return JSONResponse(content={"error": "No signs have been recognized yet"}, status_code=400)

        final_sentence = " ".join(predicted_texts)  # Combine recognized labels
        audio_file = "final_output.mp3"

        # Convert to voice
        tts = gTTS(text=final_sentence, lang='ar', slow=False)
        tts.save(audio_file)

        # Clear the stored texts after voice generation
        predicted_texts.clear()

        return FileResponse(audio_file, media_type="audio/mpeg", filename="final_output.mp3")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
