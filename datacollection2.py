import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import time

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "C:\Users\deads\PycharmProjects\GestureGennie\Data\A"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Preventing out-of-bounds
        img_crop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]
        imgCropShape = img_crop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(img_crop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            img_white[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(img_crop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            img_white[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", img_crop)
        cv2.imshow("ImageWhite", img_white)

        key = cv2.waitKey(1)
        if key == ord("Z"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', img_white)
            print(counter)

    cv2.imshow("Image", img)
