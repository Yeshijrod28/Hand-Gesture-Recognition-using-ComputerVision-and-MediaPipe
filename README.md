# hand-gesture-recognition-using-computerVision-and-MediaPipe

Predicting hand gestures in real time using your webcam. It utilizes OpenCV for image processing, MediaPipe for hand landmark detection and an LSTM (Long Short-Term Memory) neural network for gesture sequence classification.

## Features

- Real-time hand gesture recognition via webcam
- Custom dataset for three gestures:
  - **Kuzuzangpo**
  - **I Love You**
  - **Peace Sign**
- Uses OpenCV, MediaPipe, and an LSTM deep learning model

## Demo

<!-- Add your images below once you upload them -->
### Data Collection Example
![Data Collection Example](data_collection_photo.png)

### Prediction Example
![Prediction Example](prediction_photo.png)

### Prerequisites

- Python 3.8 or higher recommended
- A webcam

### Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

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

3. Run the main script (replace `main.py` with your actual entry point if it’s different):
    ```bash
    python main.py
    ```

4. Present one of the supported gestures in front of your webcam to see the prediction.

## Model Details

- **Data Collection:** Custom images were collected for three gestures: kuzuzangpo, iloveyou, and peace sign.
- **Hand Landmark Detection:** MediaPipe is used to extract hand landmarks from each video frame.
- **Gesture Classification:** An LSTM model is trained to recognize sequences of hand movements and classify them into one of the three gestures.

## Project Structure

```
hand-gesture-recognition-using-computerVision-and-MediaPipe/
├── gesture_data/                       # Collected gesture data
├── gesture_env
├── processed_data/
├── models/                     # Trained models
├── requirements.txt
└── ...
```

**Author:** Yeshijrod28  
For questions or contributions, feel free to open an issue or pull request.
