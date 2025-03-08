import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import tensorflow as tf
import math
import tensorflow as tf

# Load old model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Save in new format
model.save("new_model.keras", save_format="keras")

# Try loading the new model
classifier = tf.keras.models.load_model("new_model.keras")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

labels = ["hello", "thank you", "please", "yes"]

while True:
    success, img = cap.read()
    if not success:
        break  # If frame is not captured, exit loop

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping stays within frame
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y1:y2, x1:x2]

        # Check if image crop is valid
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Predict using latest TensorFlow model
            imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension
            prediction = classifier.predict(imgWhite)  # Get predictions
            index = np.argmax(prediction)  # Get highest probability class

            # Draw labels
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), 
                          (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), 
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), 
                          (x + w + offset, y + h + offset), (0, 255, 0), 4)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop on 'q' press

cap.release()
cv2.destroyAllWindows()
