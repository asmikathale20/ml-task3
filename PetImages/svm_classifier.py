import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = []
labels = []

dataset_path = "PetImages"

categories = ["Cat", "Dog"]

for category in categories:
    folder_path = os.path.join(dataset_path, category)

    label = categories.index(category)

    count = 0

    for image_name in os.listdir(folder_path):

        try:
            image_path = os.path.join(folder_path, image_name)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (64, 64))

            image = image.flatten()

            data.append(image)
            labels.append(label)

            count += 1

            if count == 500:
                break

        except:
            pass

X = np.array(data)
y = np.array(labels)

print("Total Images Loaded:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training SVM...")

model = SVC(kernel="linear")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy * 100)

print("\nClassification Report:")
print(classification_report(y_test, predictions))