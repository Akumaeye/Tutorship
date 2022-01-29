import cv2
import os
import numpy as np
import tensorflow as tf


def Face_Emotion_detector():
    
    # Importing the OpenCv cascade classifier for faces 
    cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    #Importing the trained model
    trained_model =  tf.keras.models.load_model('trained_models/emotion_model.h5')
    
    # Opening Webcam
    video_capture = cv2.VideoCapture(0)
    
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: 'Surprise'}
    
    while True:
         # Capture frame-by-frame
         ret, frame = video_capture.read()
         
         if not ret:
             break
    
         # Turning images from BRG to Gray 
         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
         # face detection
         faces = faceCascade.detectMultiScale(
              gray,
              scaleFactor=1.1,
              minNeighbors=5,
              minSize=(30, 30),
              flags=cv2.CASCADE_SCALE_IMAGE
          )
         # Draw a rectangle around the faces
         if ret == True:     
              for x1,y1,h,w in faces:
                  
                  x2, y2 = x1 + w, y1 + h
                  face = frame[y1:y2, x1:x2]
                  gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
                  cv2.rectangle(frame,  (x1, y1), (x2, y2), (0, 255, 255), 2)
                  
                  # Preparing the detected face for prediction (input shape)
                  cropped_img = cv2.resize(gray_face, (48,48))
                  img = np.array(cropped_img)
                  img = np.expand_dims(img, axis=0)
                  img = img.reshape((1,48,48,1))
                
                  # Make prediction on the detected face
                  prediction = trained_model.predict(img)  
                  print(prediction)
                  maxindex = int(np.argmax(prediction))
                  
                  # write the prediction above the face
                  cv2.putText(frame, emotion_dict[maxindex],(x1+20, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                         
         # Display the resulting frame
         cv2.imshow('Video', frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


Face_Emotion_detector()