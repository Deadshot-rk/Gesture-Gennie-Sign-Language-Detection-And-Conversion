# Gesture-Gennie-Sign-Language-Detection-And-Conversion
Gesture Gennie – Sign Language Gesture Recognition System
📌 Project Overview

Gesture Gennie is a real-time sign language gesture recognition system developed using Python, computer vision, and supervised machine learning. The system detects hand gestures through a webcam, processes them using image preprocessing techniques, classifies gestures using a trained model, and converts recognized gestures into readable text or speech output.

⚙️ Technologies & Tools Used

Programming Language: Python

Machine Learning: Google Teachable Machine (Supervised Image Classification)

Computer Vision & Image Processing: OpenCV, MediaPipe

Hand Tracking: MediaPipe Hands

Dataset: Custom-labeled gesture dataset (ASL-inspired hand gestures)

Development Environment: PyCharm

Text-to-Speech: Google Text-to-Speech (gTTS)

🧠 System Workflow

Captures live video feed using a webcam.

Detects and tracks hands using MediaPipe.

Crops hand regions dynamically and applies padding.

Normalizes images to a fixed 300×300 format while preserving aspect ratio.

Trains a supervised image classification model using Google Teachable Machine.

Performs real-time gesture recognition using the trained model.

Converts recognized gestures into text and speech output using Google TTS.

📂 Dataset Generation & Training

Created a custom dataset of 4,800+ labeled images (600 images per gesture).

Dataset inspired by American Sign Language (ASL) gestures.

Images captured under varying lighting conditions for better generalization.

Dataset used to train a supervised classification model via Teachable Machine.

🚀 Key Features

Real-time hand and gesture detection

Supports 8+ hand gestures

Stable performance under different lighting conditions

Gesture-to-text and gesture-to-speech conversion

Lightweight and beginner-friendly ML pipeline

▶️ How to Run
pip install opencv-python mediapipe cvzone numpy gtts
python main.py


Ensure webcam access is enabled.

📊 Results

Achieved stable real-time gesture recognition during live testing.

Successfully converted recognized gestures into text and speech output.

Demonstrated reliable performance for educational and assistive use cases.

📁 Dataset

Dataset was manually created and labeled,but a sample trained model is uploaded.

Not uploaded to GitHub due to size constraints.
