import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
from picamera2.encoders import Encoder

# Set up the camera
# 640x480 resolution
# Null encoder is being used ( i dont know if a encoder could be helpful here, need to check)
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}, raw={}, encode="raw")
picam2.configure(video_config)
encoder = Encoder() # null encoder

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# The model being used is on 16x8 mode
# The default delegate could not handle this kind of quantization, so Arm nn delegate is being used
# I need to investigate why the Arm nn delegate is returning warnings
delegate_lib = tf.lite.experimental.load_delegate('/home/rasp/Downloads/armdelegate/delegate/libarmnnDelegate.so', options={"backends": "CpuRef"})

# Load the model
interpreter = tf.lite.Interpreter(model_path='../models/models_tflite/fnet.tflite',
                                             experimental_delegates=[delegate_lib])
interpreter.allocate_tensors()

# Input details is basically the expected input shape
# Output details is the expected outcome
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Begin the recording
picam2.start_recording(encoder, "test.raw")

frame_buffer = np.empty((1, 44, 132)) # The input of the model is a 3D array
frame_data = np.empty((33,4)) # This is the structure where the x,y,z and visibility is placed during the capture of the frame
fall_buffer = [0, 0, 0] 

status = "Loading..."
color = (255, 0, 0)


while True:
    frame = picam2.capture_array() # current frame
    results = pose.process(frame) #mediapipe processing the frame

    if results.pose_landmarks:
        for i,landmark in enumerate(results.pose_landmarks.landmark): # extraction of landmarks (x,y,z and visibility)
            frame_data[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
            
        # Insertion of a new frame_data
        frame_buffer = np.roll(frame_buffer, -1, axis=1)
        frame_buffer[0, -1] = frame_data.flatten()
        frame_data.fill(0)
        
        # Is there 44 frames?
        if np.all(frame_buffer != 0):
            input_data = frame_buffer.astype(np.float32)

            # Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            sensibilidadeDaQueda = interpreter.get_tensor(output_details[0]['index'])[0][0]

            # Checks the sensitivity to falling
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

        # Draw landmarks
        visible_dots = 0
        for landmark in results.pose_landmarks.landmark:
            height, width, _ = frame.shape
            x, y = int(landmark.x * width), int(landmark.y * height)
            visibility = landmark.visibility
            if visibility < 0.6:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            elif visibility < 0.85:
                visible_dots += 1
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            else:
                visible_dots += 1
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Status
    cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Window view
    cv2.imshow("Webcam Pose Detection", frame)

    # To exit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop_recording()
cv2.destroyAllWindows()
