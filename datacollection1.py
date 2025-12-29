import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Allow detection of up to two hands

offset = 20
imgSize = 300

folder = "Data/BYE"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        if len(hands) == 2:
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']

            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
        else:
            x, y, w, h = hands[0]['bbox']

        img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop with offset and prevent out-of-bounds
        img_crop = img[max(0, y - offset): y + h + offset,
                   max(0, x - offset): x + w + offset]
        imgCropShape = img_crop.shape

        aspectRatio = imgCropShape[0] / imgCropShape[1]

        if aspectRatio > 1:
            k = imgSize / imgCropShape[0]
            wCal = math.ceil(k * imgCropShape[1])
            imgResize = cv2.resize(img_crop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            img_white[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / imgCropShape[1]
            hCal = math.ceil(k * imgCropShape[0])
            imgResize = cv2.resize(img_crop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            img_white[hGap:hGap + hCal, :] = imgResize

        cv2.imshow("ImageCrop", img_crop)
        cv2.imshow("ImageWhite", img_white)

        key = cv2.waitKey(1)
        if key == ord("P"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', img_white)
            print(counter)

    cv2.imshow("Image", img)
