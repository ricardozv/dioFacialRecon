import tensorflow as tf
import numpy as np

def load_facenet_model(model_path):
    return tf.keras.models.load_model(model_path)

def recognize_face(face_embedding, classifier):
    return classifier.predict([face_embedding])
