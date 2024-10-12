import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding, PositionEmbedding
from keras.callbacks import EarlyStopping

print('1: Transformer')
print('2: Fnet')
print('3: MLP-Mixer')
option = input("Choose a model: ")

if option == "1":
    loaded_model = keras.models.load_model('../models/trained_model_keras_nlp.keras')
elif option == "2":
    loaded_model = keras.models.load_model('../models/trained_model_FNetEcoder.keras')
elif option == "3":
    loaded_model = keras.models.load_model('../models/trained_model_MLPMixer.keras')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

frame_buffer = []
buffer_queda = [0, 0, 0]

textoAtual = "carregando"
cor = (255, 0, 0)
pontosVisiveis = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose.process(frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        frame_data = []
        for landmark in landmarks:
            x, y, z, visibility = landmark.x, landmark.y, landmark.z, landmark.visibility
            frame_data.extend([x, y, z, visibility])
        frame_buffer.append(frame_data)

        if len(frame_buffer) > 44:
            frame_buffer.pop(0)
            chanceQueda = loaded_model.predict(np.array(frame_buffer).reshape(1, 44, 132))

            #if chanceQueda > 0.99:
            if False:
                buffer_queda.pop(0)
                buffer_queda.append(1)
                if np.sum(buffer_queda) == len(buffer_queda):
                    textoAtual = "FALL!"
                    cor = (150, 150, 255)
            else:
                buffer_queda.pop(0)
                buffer_queda.append(0)
                textoAtual = "Stabilized"
                cor = (0, 255, 0)

            pontosVisiveis = 0
            for landmark in results.pose_landmarks.landmark:
                height, width, _ = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                visibility = landmark.visibility
                if visibility < 0.6:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                elif visibility < 0.85:
                    pontosVisiveis += 1
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                else:
                    pontosVisiveis += 1
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            
            if pontosVisiveis < 0:
                textoAtual = "Pontos insuficientes"
                cor = (255, 255, 0)

    height, width, _ = frame.shape
    #cv2.rectangle(frame, (0,0), (width,30), (255, 255, 255), -1,)
    cv2.rectangle(frame, (0,0), (width,40), (0, 0, 0), -1,)

    #cv2.putText(frame, "Detector de Quedas com Inteligencia Artificial", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, textoAtual, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
    cv2.putText(frame, "| Pontos: " + str(pontosVisiveis), (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    cv2.namedWindow("Webcam Pose Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Webcam Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Webcam Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
