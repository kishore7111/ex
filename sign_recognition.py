import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Image size and dataset path
IMG_SIZE = 64
DATASET_PATH = 'C:/SSD/BIT/S5-LAB/ML/EXP5/dataset'

# Load the dataset
def load_data(dataset_path):
    images = []
    labels = []
    classes = os.listdir(dataset_path)

    for label in classes:
        for img_file in os.listdir(os.path.join(dataset_path, label)):
            img_path = os.path.join(dataset_path, label, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(classes.index(label))

    return np.array(images), np.array(labels), classes

# Load images and labels
images, labels, class_names = load_data(DATASET_PATH)
images = images / 255.0  # Normalize images

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
def create_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create and compile the model
model = create_model(len(class_names))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save('sign_language_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Load the trained model
model = tf.keras.models.load_model('sign_language_model.h5')

# Real-time sign language recognition using webcam
cap = cv2.VideoCapture(0)  # Capture video from webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Display prediction on frame
    cv2.putText(frame, f'Sign: {class_names[class_index]} - {confidence:.2f}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
