# Create Data from images that collect by 'collectImages.py'

import os
import pickle

import mediapipe as mp
import cv2
# import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands =mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5)

DATA_DIR = './images'

# data & labels will be stored to pickle file
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    
    # Extract only directories under DATA_DIR
    directories = [directory for directory in dir_ if os.path.isdir(os.path.join(DATA_DIR, dir_))]

    # Print the list of directories
    for directory in directories:
        # print('directory',directory)
        for imgPath in os.listdir(os.path.join(DATA_DIR, directory)):
            print('imgPath',imgPath)
            # temporary list used to store the normalized coordinates of landmarks
            dataAux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, imgPath))
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        dataAux.append(x - min(x_))
                        dataAux.append(y - min(y_))

                data.append(dataAux)
                labels.append(dir_)

with open('images.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
    
    