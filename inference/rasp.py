import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2, Preview

# Set up the camera
#640x480 resolution
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888"}))

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=0)

frame_buffer = np.zeros((1, 44, 132)) #The input of the model is a 3D array
frame_data = np.zeros((33,4)) #Receives the x,y,z and visibility of the frame
fall_buffer = [0, 0, 0] 

status = "Loading..."
color = (255, 0, 0)

# Load the model
interpreter = tf.lite.Interpreter(model_path='../models/models_tflite/fnet_float_quantization.tflite')
interpreter.allocate_tensors()

#Input details is basically the expected input shape
#Output details is the expected outcome
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Configure drawing styles
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

picam2.start()

while True:
    frame = picam2.capture_array() #current frame
    results = pose.process(frame) #mediapipe processing 

    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark): #extraction of landmarks
            frame_data[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        
        #Insertion of a new frame data
        frame_buffer = np.roll(frame_buffer, -1, axis=1) 
        frame_buffer[0, -1] = frame_data.flatten()
        
        #frame_buffer is filled with landmarks?
        if np.all(frame_buffer != 0):
            input_data = frame_buffer.astype(np.float32)
            
            #Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            sensibilidadeDaQueda = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            if sensibilidadeDaQueda > 0.97:
                fall_buffer.pop(0)
                fall_buffer.append(1)
                if np.sum(fall_buffer) == len(fall_buffer):
                    status = "Fall!"
                    color = (0, 0, 255)
            else:
                fall_buffer.pop(0)
                fall_buffer.append(0)
                status = "Stabilized"
                color = (0, 255, 0)
        
        # Draw landmarks using MediaPipe's built-in function
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )
    
    # Status
    cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Window view
    cv2.imshow("Webcam Pose Detection", frame)

    # To exit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        picam2.stop()
        break

cv2.destroyAllWindows()