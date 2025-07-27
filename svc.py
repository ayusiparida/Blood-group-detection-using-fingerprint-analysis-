# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:44:05 2024

@author: AYUSI
"""

import os
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle

# Prepare data
img2vec = Img2Vec()

# Directory where all 8 blood group folders are stored
dataset_dir = r"C:\fingerprint\dataset_blood_group"
features = []
labels = []

# Check how many images are in each category (blood group) folder
for category in os.listdir(dataset_dir):
    category_dir = os.path.join(dataset_dir, category)
    num_images = len(os.listdir(category_dir))
    print(f'Blood group "{category}" has {num_images} images.')

    # If there are too few images, flag the category
    if num_images == 0:
        print(f'Warning: Blood group "{category}" has no images.')
    
    # Process images for each blood group
    for img_path in os.listdir(category_dir):
        img_path_ = os.path.join(category_dir, img_path)
        
        try:
            img = Image.open(img_path_)

            # Resize and convert to tensor without normalization for grayscale images
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),  # Convert to RGB
                transforms.ToTensor(),
            ])

            img_tensor = preprocess(img)

            # Get features using img2vec
            img_pil = transforms.ToPILImage()(img_tensor)
            img_features = img2vec.get_vec(img_pil)

            features.append(img_features)
            labels.append(category)
        
        except Exception as e:
            print(f"Error processing {img_path_}: {e}")
            continue

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Check if there are enough samples for training
if len(features) == 0 or len(labels) == 0:
    print("Error: No data found. Please check your dataset.")
else:
    print(f"Total images: {len(features)}")
    
    # Use all data for training (no train-test split due to small dataset)
    train_data = features
    train_labels = labels

    # Train SVC model
    svc_model = SVC(kernel='linear', C=0.2, random_state=2)
    svc_model.fit(train_data, train_labels)

    # Predict on the training data itself (since we have no validation set)
    y_pred_svc = svc_model.predict(train_data)
    score_svc = accuracy_score(y_pred_svc, train_labels)
    print('SVC model training accuracy is:', score_svc)

    # Calculate individual accuracy for each blood group
    blood_group_labels = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']
    blood_group_accuracy = {}
    for blood_group in blood_group_labels:
        # Filter training data and labels for the current blood group
        blood_group_indices = train_labels == blood_group
        training_data_blood_group = train_data[blood_group_indices]
        training_labels_blood_group = train_labels[blood_group_indices]

        # Predict using SVC model
        y_pred_svc_blood_group = svc_model.predict(training_data_blood_group)
        accuracy_svc_blood_group = accuracy_score(y_pred_svc_blood_group, training_labels_blood_group)
        blood_group_accuracy[f'SVC_{blood_group}'] = accuracy_svc_blood_group

    # Print individual accuracies
    print('\nIndividual Accuracies:')
    for blood_group, accuracy in blood_group_accuracy.items():
        print(f'{blood_group}: {accuracy}')

    # Calculate and print classification report for overall model
    print('\nClassification Report - SVC:')
    print(classification_report(train_labels, y_pred_svc, target_names=blood_group_labels))

    # Plot confusion matrix
    cf = confusion_matrix(train_labels, y_pred_svc)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=blood_group_labels)
    cmd.plot()
    plt.show()

    
