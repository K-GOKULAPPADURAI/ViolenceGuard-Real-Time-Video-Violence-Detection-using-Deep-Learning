import cv2
import numpy as np
import os
from keras.models import load_model

# Load the model
model = load_model('model2.h5')
# Load the video file

#video_path = "noFight/nofi007.mp4"
video_path = "fight/fi001.mp4"

video = cv2.VideoCapture(video_path)

# Create a function to preprocess the frame
def preprocess(frame):
    # Resize the frame if needed
    # processed_frame = cv2.resize(frame, (width, height))
    
    # Preprocess the frame (e.g., resize, normalize, etc.)
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = normalized_frame.reshape(1, 224, 224, 3)
    return input_frame

# Create a function to process and display the video
def process_video(video):
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        # Preprocess the frame
        processed_frame = preprocess(frame)

        # Make a prediction on the processed frame
        prediction = model.predict(processed_frame)

        # Determine the class label based on the prediction
        if prediction[0][0] >= 0.5:
            label = 'Fighting recorded'
            print('Fighting recorded ',prediction[0][0]*100,'%')
        else:
            label = 'No fight'
            print('No fight ',prediction[0][1]*100,'%')
            

        # Overlay the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video object
    video.release()
    cv2.destroyAllWindows()

# Process and display the video
process_video(video)
