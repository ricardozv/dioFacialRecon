from sklearn.svm import SVC
import numpy as np

def train_classifier(embeddings, labels):
    classifier = SVC(kernel='linear')
    classifier.fit(embeddings, labels)
    return classifier
