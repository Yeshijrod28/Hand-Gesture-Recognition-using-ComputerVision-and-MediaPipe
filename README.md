# hand-gesture-recognition-using-computerVision-and-MediaPipe

Predicting hand gestures in real time using your webcam. It utilizes OpenCV for image processing, MediaPipe for hand landmark detection and an LSTM (Long Short-Term Memory) neural network for gesture sequence classification.

---
## Project Overview

The system have these main parts:
- Data Collection: Captures hand gestures via webcam using OpenCV
- Preprocessing: Extracts hand landmarks using MediaPipe, normalize and sequence the data
- Model Training: Build and train an LSTM model using TensorFlow/Keras
- Prediction & Output: Real-time gesture recognition with text-to-speech conversion

---
## Features
- Real-time hand gesture recognition via webcam
- Custom dataset for three gestures:
  - **Kuzuzangpo** 👋
  - **I Love You** 🤟
  - **Peace Sign** ✌️
- Uses OpenCV, MediaPipe, and an LSTM deep learning model
- Uses pyttsx3 to convert detected gestures into speech
- Pandas and Matplotlib for visualization
---
### Data Collection 
 ![Screenshot 2025-06-24 011036](https://github.com/user-attachments/assets/3e22542e-749a-40a2-b347-8a1869c060b9)
 ![Screenshot 2025-06-24 010949](https://github.com/user-attachments/assets/27640571-a218-4ed3-86a8-d14dd1c20f73)
---
## Demo Prediction Example
- https://drive.google.com/file/d/1NdPz_-Bkr3fchtDG7XSSbD_2wU13-_ID/view?usp=sharing
---
### Prerequisites

- Python 3.8 or higher recommended
- A webcam

**Key dependencies:**
- opencv-python
- mediapipe
- tensorflow
- numpy
- scikit-learn
- matplotlib

(See `requirements.txt` for the full list.)

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/Yeshijrod28/hand-gesture-recognition-using-computerVision-and-MediaPipe.git
    cd hand-gesture-recognition-using-computerVision-and-MediaPipe
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the .ipynb file
   
4. Run the main script:
    ```bash
    python main.py
    ```

5. Present one of the supported gestures in front of your webcam to see the prediction.

## Model Details

- **Data Collection:** Custom images were collected for three gestures: kuzuzangpo, iloveyou, and peace sign.
- **Hand Landmark Detection:** MediaPipe is used to extract hand landmarks from each video frame.
- **Gesture Classification:** An LSTM model is trained to recognize sequences of hand movements and classify them into one of the three gestures.
  
- ![Screenshot 2025-06-24 023500](https://github.com/user-attachments/assets/1cbf829c-a405-4789-8b3c-d4c1fefdff7a)


## Project Structure

```
hand-gesture-recognition-using-computerVision-and-MediaPipe/
├── gesture_data/                       # Collected gesture data
├── gesture_env
├── models/                     # Trained models
├── processed_data/
├── main.py
├── requirements.txt
└── ...
```

