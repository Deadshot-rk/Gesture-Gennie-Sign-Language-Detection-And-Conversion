import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Load trained classifier model
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants
offset = 20
imgSize = 300

# Load labels from file (only names, not indexes)
with open("Model/labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        # If both hands are present, use bounding box covering both
        if len(hands) == 2:
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']

            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
        else:
            x, y, w, h = hands[0]['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255

        # Crop hand region from original image
        imgCrop = img[max(0, y - offset): y + h + offset,
                      max(0, x - offset): x + w + offset]

        imgCropShape = imgCrop.shape
        aspectRatio = imgCropShape[0] / imgCropShape[1]

        # Resize to square white background
        if aspectRatio > 1:
            k = imgSize / imgCropShape[0]
            wCal = math.ceil(k * imgCropShape[1])
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / imgCropShape[1]
            hCal = math.ceil(k * imgCropShape[0])
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Prepare image for model: resize + normalize
        imgInput = cv2.resize(imgWhite, (224, 224))
        imgInput = imgInput.astype('float32') / 255.0

        # Predict
        prediction, index = classifier.getPrediction(imgInput, draw=False)

        # Display prediction
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x + w + offset, y - offset), (240, 248, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 28),
                    cv2.FONT_ITALIC, 1.5, (33, 37, 41), 3)

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (220, 235, 255), 4)

        # Show cropped and white images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
