# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:32:45 2025

@author: ayush
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from torchvision import transforms
from img2vec_pytorch import Img2Vec
import pickle

# Initialize Img2Vec with pretrained ResNet18
img2vec = Img2Vec()

# Path to your dataset (folder containing subfolders like A+, B-, etc.)
dataset_dir = r"C:\Users\ayush\OneDrive\Desktop\project expo 1\dataset_blood_group"
features = []
labels = []

# Process images from each subfolder (label = folder name)
for label in os.listdir(dataset_dir):
    label_path = os.path.join(dataset_dir, label)
    if os.path.isdir(label_path):
        image_files = os.listdir(label_path)
        print(f'Label "{label}" has {len(image_files)} images.')

        for img_file in image_files:
            img_path = os.path.join(label_path, img_file)
            try:
                img = Image.open(img_path)

                # Preprocessing for ResNet
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Grayscale(num_output_channels=3),  # Convert to RGB shape
                    transforms.ToTensor(),
                ])

                img_tensor = preprocess(img)
                img_pil = transforms.ToPILImage()(img_tensor)

                # Extract feature vector
                vec = img2vec.get_vec(img_pil)

                features.append(vec)
                labels.append(label)

            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
                continue
    else:
        print(f"Skipping file: {label_path}")

# Convert to arrays
features = np.array(features)
labels = np.array(labels)

# Check if data exists
if len(features) == 0:
    print("Error: No data found. Please check your dataset.")
    exit()

# Train SVM
model = SVC(kernel='linear', C=0.2, random_state=42)
model.fit(features, labels)

# Predict on training data (no split here)
predictions = model.predict(features)

# Evaluation
accuracy = accuracy_score(labels, predictions)
f1_macro = f1_score(labels, predictions, average='macro')
f1_weighted = f1_score(labels, predictions, average='weighted')

print(f"\nOverall Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {f1_macro:.4f}")
print(f"Weighted F1 Score: {f1_weighted:.4f}")

# Per-class report
label_names = sorted(list(set(labels)))
print("\nClassification Report:")
print(classification_report(labels, predictions, target_names=label_names))

# Confusion Matrix
cm = confusion_matrix(labels, predictions, labels=label_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

