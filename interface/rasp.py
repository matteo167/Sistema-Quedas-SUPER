import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2

# Carrega o modelo TFLite
model_path = '../models/model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Pega detalhes do modelo TFLite
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Inicializa a câmera usando Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

frame_buffer = []
buffer_queda = [0, 0, 0]

textoAtual = "Carregando"
cor = (255, 0, 0)

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)  # Inverte a imagem horizontalmente

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
            input_data = np.array(frame_buffer).reshape(1, 44, 132).astype(np.float32)

            # Faz a predição com o modelo TFLite
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            sensibilidadeDaQueda = interpreter.get_tensor(output_details[0]['index'])[0][0]

            # Verifica a sensibilidade da queda
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

            # Desenha os pontos de pose
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

    # Exibe o status
    cv2.putText(frame, textoAtual, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

    # Janela de visualização
    cv2.imshow("Webcam Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
