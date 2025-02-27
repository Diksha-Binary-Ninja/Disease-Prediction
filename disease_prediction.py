import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the disease dataset
df = pd.read_csv(r"C:\Users\nkoni\Downloads\dataset (1).csv")
df.fillna(0, inplace=True)
df = df.drop(['Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14',
       'Symptom_15', 'Symptom_16', 'Symptom_17'], axis=1)

# Prepare symptom dictionary
unique_symptoms = df.iloc[:, 1:].stack().unique()
symptom_dict = {symptom: i for i, symptom in enumerate(unique_symptoms, 1)}
df.iloc[:, 1:] = df.iloc[:, 1:].replace(symptom_dict)

# Prepare disease dictionary
unique_diseases = df['Disease'].unique()
disease_dict = {disease: i for i, disease in enumerate(unique_diseases, 200)}
df['Disease'] = df['Disease'].replace(disease_dict)

# Train a Decision Tree model
X = df.drop('Disease', axis=1)
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

def predict_disease(symptoms):
    input_vector = np.zeros((1, X.shape[1]))
    for symptom in symptoms:
        if symptom in symptom_dict:
            input_vector[0, list(X.columns).index(symptom)] = symptom_dict[symptom]
    prediction = clf.predict(input_vector)
    predicted_disease = [k for k, v in disease_dict.items() if v == prediction[0]]
    return predicted_disease[0] if predicted_disease else "Unknown Disease"

# Dataset Path
dataset_path = r"C:\Users\nkoni\Desktop\Kidney_Dataset"

# Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=20, 
                                   zoom_range=0.2, 
                                   shear_range=0.2, 
                                   horizontal_flip=True, 
                                   validation_split=0.2)

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    dataset_path, 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='categorical', 
    subset='training')

# Load Validation Data
val_generator = train_datagen.flow_from_directory(
    dataset_path, 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='categorical', 
    subset='validation')

# Get Class Labels
class_labels = list(train_generator.class_indices.keys())
print("Class Labels:", class_labels)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes: Normal, Cyst, Stone, Tumor
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_generator, epochs=20, validation_data=val_generator)

# Save Model for Future Use
model.save('kidney_cnn_model.h5')
print("Model Saved Successfully!")

# Plot Accuracy and Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Load Trained Model
model = tf.keras.models.load_model('kidney_cnn_model.h5')

def predict_kidney_disease(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return class_labels[np.argmax(prediction)]

# Example Prediction
image_path = r"C:\Users\nkoni\Desktop\kidney_xray.jpg"
print("Kidney X-ray Prediction:", predict_kidney_disease(image_path))

