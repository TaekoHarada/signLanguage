import os
import cv2
import mediapipe as mp

windowName = 'myCam'
camWidth = 640
camHeight =360

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)
cam.set(cv2.CAP_PROP_FPS, 1)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Collect data for training
DATA_DIR = './images'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

signList=['A','B','C']

# The number of collecting images
collectDataNum = 100

for j in range(len(signList)):
    if not os.path.exists(os.path.join(DATA_DIR, signList[j])):
        os.makedirs(os.path.join(DATA_DIR, signList[j]))

    print('Collecting data for sign {}'.format(signList[j]))

    if not cam.isOpened():
        print('Cannot open camera')
        exit()

    while True:
        ret, frame = cam.read()
        cv2.putText(frame, 'Press "Q" to start collecting data.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xff ==ord('q'):
            break

    counter = 0
    while counter < collectDataNum:
        ret, frame = cam.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, signList[j], '{}.jpg'.format(counter)), frame)

        counter += 1

cam.release()
cv2.destroyAllWindows()
