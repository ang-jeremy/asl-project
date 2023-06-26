import mediapipe as mp
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import text

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # recolour feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # make detections
        results = holistic.process(image)

        # recolour image back BGR for renderings
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # draw hand landmarks for left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # draw hand landmarks for right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('webcam feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

