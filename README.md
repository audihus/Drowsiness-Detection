# Drowsiness Detection

Desktop application for real-time drowsiness detection using a webcam. The
application detects a face with dlib, extracts facial landmarks, and calculates
the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to identify prolonged
eye closure and yawning.

## Features

- Real-time webcam processing
- Facial-landmark detection with dlib
- Drowsiness alert based on eye closure and yawning
- Audible alarm when the configured threshold is reached
- Experimental custom landmark model for eyes and mouth

## Requirements

Install the Python packages used by the application:

```bash
pip install dlib opencv-python imutils scipy playsound
```

## Run the application

```bash
python main.py
```

The application uses the default webcam (`WEBCAM_INDEX = 0`). Change that
setting in `main.py` if another camera should be used.

## Models

The baseline dlib model used by the original application is
`shape_predictor_68_face_landmarks.dat`.

A custom eye-and-mouth landmark model trained for this project is available in
[`models/eye-mouth-landmarks`](models/eye-mouth-landmarks). It is stored with
Git LFS because each model file is larger than 50 MB. After cloning, retrieve
the model files with:

```bash
git lfs pull
```

## Paper

The project paper is available at
[`docs/Real-Time-Drowsiness-Detection-Based-on-Facial-Features.pdf`](docs/Real-Time-Drowsiness-Detection-Based-on-Facial-Features.pdf).

## Project structure

```text
.
├── main.py                         # Desktop application entry point
├── shape_predictor_68_face_landmarks.dat
├── alertSound.wav
├── models/eye-mouth-landmarks/      # Custom dlib models and model card
└── docs/                            # Project paper
```
