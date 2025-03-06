import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding, PositionEmbedding



# print('1: Transformer')
# print('2: Fnet')
# print('3: MLP-Mixer')
# option = input("Choose a model: ")

option = "1"

if option == "1":
    loaded_model = keras.models.load_model('../models/trained_model.keras')
elif option == "2":
    loaded_model = keras.models.load_model('../models/trained_model_FNetEcoder.keras')
elif option == "3":
    loaded_model = keras.models.load_model('../models/trained_model_MLPMixer.keras')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

frame_buffer = []
buffer_queda = [0,0,0]

textoAtual = "carregando"
cor = (255, 0, 0)

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
            sensibilidadeDaQueda = loaded_model.predict(np.array(frame_buffer).reshape(1, 44, 132))
            #print(sensibilidadeDaQueda)
            if sensibilidadeDaQueda > 0.99:
                buffer_queda.pop(0)
                buffer_queda.append(1)
                if np.sum(buffer_queda) == len(buffer_queda):
                    textoAtual = "Queda!"
                    cor = (0, 0, 255)
            else:
                buffer_queda.pop(0)
                buffer_queda.append(0)
                textoAtual = "Estabilizado"
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
            
            # print(pontosVisiveis)
            # if pontosVisiveis < 25:
            #     textoAtual = "Pontos insuficientes"
            #     cor = (255, 255, 0)


    cv2.putText(frame, textoAtual, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)


    cv2.namedWindow("Webcam Pose Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Webcam Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Webcam Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()   