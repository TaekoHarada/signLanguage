import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Get only 1 hand
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=1)

# labels_dict = {0: 'A', 1: 'B', 2: 'C'}

while True:

    dataAux = []
    x_ = []
    y_ = []

    ret, frame = cam.read()

    fHeight, fWidth, _ = frame.shape

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frameRGB)
    
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                handLandmarks,
                mp_hands.HAND_CONNECTIONS)

            for i in range(len(handLandmarks.landmark)):
                x = handLandmarks.landmark[i].x
                y = handLandmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(handLandmarks.landmark)):
                x = handLandmarks.landmark[i].x
                y = handLandmarks.landmark[i].y
                dataAux.append(x - min(x_))
                dataAux.append(y - min(y_))

        x1 = int(min(x_) * fWidth) - 10
        y1 = int(min(y_) * fHeight) - 10

        x2 = int(max(x_) * fWidth) - 10
        y2 = int(max(y_) * fHeight) - 10

        prediction = model.predict([np.asarray(dataAux)])
        print('prediction', prediction)

        predicted_character = prediction[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
