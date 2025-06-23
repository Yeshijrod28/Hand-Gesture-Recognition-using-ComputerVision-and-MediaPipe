import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import pyttsx3
import mediapipe as mp
import joblib 

# === Load model and label encoder ===
model = tf.keras.models.load_model('./models/best_gesture_model.h5')
label_encoder = joblib.load('./processed_data/label_encoder.pkl') 

# === Initialize MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Text-to-Speech ===
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)
last_spoken = ""

# === Real-time prediction ===
def real_time_prediction():
    global last_spoken
    cap = cv2.VideoCapture(0)
    sequence_buffer = deque(maxlen=30)

    print("Starting real-time gesture recognition with TTS...")
    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        prediction_text = "No hand detected"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                wrist_x, wrist_y = landmarks[0], landmarks[1]
                normalized_landmarks = landmarks.copy()
                for i in range(0, len(landmarks), 3):
                    normalized_landmarks[i] -= wrist_x
                    normalized_landmarks[i+1] -= wrist_y

                sequence_buffer.append(normalized_landmarks)

                if len(sequence_buffer) == 30:
                    sequence = np.array(list(sequence_buffer))
                    sequence = np.expand_dims(sequence, axis=0)

                    predictions = model.predict(sequence, verbose=0)
                    confidence = np.max(predictions)
                    predicted_index = np.argmax(predictions)
                    predicted_gesture = label_encoder.inverse_transform([predicted_index])[0]

                    if confidence > 0.7:
                        prediction_text = f"{predicted_gesture} ({confidence:.2f})"

                        if predicted_gesture != last_spoken:
                            tts_engine.say(predicted_gesture)
                            tts_engine.runAndWait()
                            last_spoken = predicted_gesture
                    else:
                        prediction_text = "Uncertain"

        cv2.putText(frame, f'Prediction: {prediction_text}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Buffer: {len(sequence_buffer)}/30', 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Real-time Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Run the app ===
if __name__ == "__main__":
    real_time_prediction()
