import cv2
import numpy as np
from keras.models import model_from_json
from tkinter import *

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

def start_capture():
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture", frame)
        k = cv2.waitKey(1)

        # take screenshot with 's' key
        if k%256 == 115:
            filename = "capture_{}.jpg".format(count)
            cv2.imwrite(filename, frame)
            print("{} saved".format(filename))
            count += 1

        # stop video capture with 'q' key
        elif k%256 == 113:
            break

    cap.release()
    cv2.destroyAllWindows()

    pass

def start_recognition():
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

root = Tk()  # create root window
root.title("Face Emotion Recognition")
root.config(bg="skyblue")

# Create Frame widget
left_frame = Frame(root, width=1024, height=640)
left_frame.grid(row=0, column=0, padx=80, pady=40)

# Create Label widget
Label(left_frame, text="Face Emotion Recognition", font=("Helvetica", 20)).grid(row=0, column=0, padx=5, pady=5)

# Create Capture Video Label widget
Label(left_frame, text="Capture Video:", font=("Helvetica", 16)).grid(row=1, column=0, padx=5, pady=5)

# Create Video Capture Button widget
Button(left_frame, text="Start Capture", font=("Helvetica", 14), command=start_capture).grid(row=2, column=0, padx=5, pady=5)

# Create Recognize Emotion Label widget
Label(left_frame, text="Recognize Emotion:", font=("Helvetica", 16)).grid(row=3, column=0, padx=5, pady=5)

# Create Emotion Recognition Button widget
Button(left_frame, text="Start Recognition", font=("Helvetica", 14), command=start_recognition).grid(row=4, column=0, padx=5, pady=5)

root.mainloop()
