import cv2
import mediapipe as mp
import pandas as pd
import os
import sys

# Função para processar vídeos com diferentes modelos MediaPipe Pose
def process_video(video_path, output_folder, model_name, model_complexity):
    # Inicializar MediaPipe Pose
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        cap = cv2.VideoCapture(video_path)

        data_normalized = []
        data_world = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Conversão para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Processar com MediaPipe Pose
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Obter pontos-chave normalizados com visibilidade
                normalized_keypoints = [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ]
                data_normalized.append([coord for point in normalized_keypoints for coord in point])

                # Obter pontos-chave no mundo real com visibilidade
                if results.pose_world_landmarks:
                    world_keypoints = [
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in results.pose_world_landmarks.landmark
                    ]
                    data_world.append([coord for point in world_keypoints for coord in point])

                # Desenhar os pontos sobrepostos no vídeo
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # Mostrar o vídeo com pontos sobrepostos
            cv2.imshow("Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

        # Salvar dados em CSV
        columns = [f"{axis}_{i}" for i in range(33) for axis in ["x", "y", "z", "visibility"]]
        if data_normalized:
            pd.DataFrame(data_normalized, columns=columns).to_csv(
                os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_{model_name}_normalized.csv"), index=False
            )
        if data_world:
            pd.DataFrame(data_world, columns=columns).to_csv(
                os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_{model_name}_world.csv"), index=False
            )

# Verificar argumentos da linha de comando
if len(sys.argv) < 3:
    print("Uso: python script.py <video_folder> <output_folder>")
    sys.exit(1)

video_folder = sys.argv[1]  # Primeiro argumento: pasta de vídeos
output_folder = sys.argv[2]  # Segundo argumento: pasta de saída
os.makedirs(output_folder, exist_ok=True)

# Processamento dos vídeos
models = {
    "lite": 0,
    "full": 1,
    "heavy": 2,
}

for video_file in os.listdir(video_folder):
    if video_file.endswith(".avi"):
        video_path = os.path.join(video_folder, video_file)

        # Configurar janela para tela cheia
        cv2.namedWindow("Pose Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for model_name, model_complexity in models.items():
            process_video(video_path, output_folder, model_name, model_complexity)

cv2.destroyAllWindows()
