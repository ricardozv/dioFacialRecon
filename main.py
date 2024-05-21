import cv2
import dlib
import tensorflow as tf
import numpy as np
from sklearn.svm import SVC

from src.data_preprocessing import load_and_preprocess_images
from src.face_detection import detect_faces
from src.face_recognition import load_facenet_model, recognize_face
from src.train_classifier import train_classifier

# Carregar dados e modelos
images = load_and_preprocess_images('data/raw')
facenet_model = load_facenet_model('models/facenet_model.h5')
classifier = train_classifier(embeddings, labels)  # embeddings e labels devem ser definidos

# Captura de vídeo em tempo real
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        face_embedding = facenet_model.predict(face_img)
        label = recognize_face(face_embedding, classifier)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
