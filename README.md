# AI Face Recognition, Emotion Detection & Attendance System

A real-time AI-based system that detects faces, recognizes registered people, estimates facial emotions, and automatically records attendance in a CSV file using a webcam.

## Features

* Real-time face detection using **YuNet**
* Face recognition using **SFace**
* Recognition of multiple registered people
* Facial emotion estimation
* Automatic attendance recording
* Daily duplicate-attendance prevention
* CSV-based attendance storage
* Dynamic loading of registered faces from the `known_faces` folder

## Technologies Used

* Python
* OpenCV
* NumPy
* ONNX models
* YuNet
* SFace
* MobileFaceNet

## Project Structure

```text
AI_Face_Emotion/
│
├── attendance/
│   └── attendance_emotion.csv
│
├── known_faces/
│   ├── Abubakar/
│   │   └── 1.jpg
│   ├── Hamza/
│   │   └── hamza.jpg
│   └── Daniyal/
│       └── 1.jpg
│
├── models/
│   ├── face_detection_yunet_2026may.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── facial_expression_recognition_mobilefacenet_2022july.onnx
│
├── .vscode/
├── attendance_emotion.py
├── load_faces.py
└── requirements.txt
```

## Installation

### 1. Create Conda Environment

```bash
conda create -n faceai python=3.12
```

Activate the environment:

```bash
conda activate faceai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Add Registered Faces

Create a separate folder for each person inside `known_faces`.

Example:

```text
known_faces/
├── Abubakar/
│   └── 1.jpg
├── Hamza/
│   └── hamza.jpg
└── Daniyal/
    └── 1.jpg
```

The folder name is used as the person's name.

## Run the Application

Activate the environment:

```bash
conda activate faceai
```

Then run:

```bash
python attendance_emotion.py
```

The webcam will start automatically.

Press:

```text
Q
```

to close the application.

## Attendance

When a registered person is recognized, the system records:

* Name
* Date
* Time
* Detected emotion

Attendance is saved in:

```text
attendance/attendance_emotion.csv
```

Example:

```csv
Name,Date,Time,Emotion
Abubakar,2026-10-03,04:33:08,sad
Hamza,2026-10-03,04:35:12,happy
```

The system prevents the same person from being marked multiple times on the same day.

## How It Works

The system follows this pipeline:

```text
Webcam
   ↓
Face Detection
   ↓
Face Alignment
   ↓
Face Recognition
   ↓
Identify Person
   ↓
Emotion Detection
   ↓
Attendance Recording
   ↓
CSV File
```

### Face Detection

YuNet detects faces from the webcam frame.

### Face Recognition

SFace generates a face embedding and compares it with the registered face embeddings.

### Emotion Detection

The facial expression model estimates one of the following emotions:

```text
Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral
```

### Attendance

If the person is successfully recognized, their attendance is stored in the CSV file.

## Important Note

Emotion detection is an AI-based **estimate** and should not be considered a guaranteed measurement of a person's actual emotional state.

## Future Improvements

Possible future improvements include:

* Web-based dashboard
* Database integration
* Better attendance reports
* Multiple images per person
* Admin panel
* User interface
* Cloud deployment
* Real-time attendance dashboard
* Improved emotion classification

## Project Goal

This project demonstrates the practical use of computer vision and deep learning models to build a real-world AI application combining:

**Face Detection + Face Recognition + Emotion Detection + Attendance Automation**
