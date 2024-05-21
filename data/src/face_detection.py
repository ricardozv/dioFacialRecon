import dlib

def detect_faces(image):
    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 1)
    return faces
