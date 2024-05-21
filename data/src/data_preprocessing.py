import cv2
import os

def load_and_preprocess_images(image_dir):
    images = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            img = cv2.imread(os.path.join(image_dir, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (160, 160))
            images.append(img)
    return images
