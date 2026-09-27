import os
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from skimage.feature import hog

import joblib


DATASET_PATH = "dataset/aug"
IMG_SIZE = (128, 128)

images = []
labels = []



# Read Dataset


for folder_name in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.isdir(folder_path):
        continue

    print("Reading:", folder_name)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        try:

            image = Image.open(image_path).convert("RGB")
            image = image.resize(IMG_SIZE)

            image_array = np.array(image)

            # Convert RGB image to grayscale
            gray_image = np.mean(image_array, axis=2)

            # Extract HOG features
            features = hog(
                gray_image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2)
            )

            images.append(features)
            labels.append(folder_name)

        except Exception as e:

            print("Skipped:", image_name)


X = np.array(images)
y = np.array(labels)


print("\nTotal images:", len(X))
print("Feature shape:", X.shape)
print("Classes:", np.unique(y))


# Encode Labels

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining SVM model...")


# SVM Model

model = SVC(
    kernel="rbf",
    probability=False,
    random_state=42
)

model.fit(X_train, y_train)


# Accuracy

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:", accuracy * 100, "%")


# Save Model

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/plant_classifier.pkl"
)

joblib.dump(
    encoder,
    "model/label_encoder.pkl"
)


print("\nModel saved successfully!")

print("Model type: HOG + SVM")
