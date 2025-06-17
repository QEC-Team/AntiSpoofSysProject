import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Directory containing folders of images categorized by person's name
dataset_dir = 'Face_Dataset'

# Initialize face analysis application
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Read images and extract features
X = []  # Feature list
y = []  # Label list

# Loop through each person's folder
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_dir):  # Check if it's a directory
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if the image file is not readable
            faces = app.get(img)
            if faces:
                face = faces[0]  # Assuming one face per image for simplicity
                X.append(face.normed_embedding)
                y.append(person_name)
            else:
                print(f"No faces detected in {img_path}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train a classifier
classifier = SVC(kernel='linear', probability=True)

# Convert X to a numpy array and ensure it's a 2D array for sklearn
X = np.array(X)
if len(X.shape) == 1 and X.size > 0:
    X = X.reshape(-1, 1)  # Reshape X to 2D array (when X is a flat array with features)

# Fit the model only if there is sufficient data
if X.size > 0 and len(y_encoded) > 0:
    classifier.fit(X, y_encoded)
    # Save the trained model and label encoder
    joblib.dump(classifier, 'face_recognizer.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("Model and label encoder have been saved.")
else:
    print("Insufficient data for training. Check your dataset.")
