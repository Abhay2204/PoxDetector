import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# Ensure the dataset directory and categories are correct
dataset_path = "MachineLearning/dataset/"
categories = ["Chickenpox", "Measles", "Monkeypox", "Normal"]

# Check if dataset and categories exist
for category in categories:
    category_path = os.path.join(dataset_path, category)
    if not os.path.exists(category_path):
        raise FileNotFoundError(f"Error: {category_path} does not exist or is empty!")

# Parameters
batch_size = 32
img_size = (224, 224)  # Higher resolution for pretrained models

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,  # Split for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training data generator
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

# Validation data generator
val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Compute class weights
classes = np.array(list(train_gen.class_indices.keys()))  # Convert to numpy array
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),  # Unique class indices
    y=train_gen.classes
)
class_weights = {i: weight for i, weight in enumerate(class_weights)}

# Load Pretrained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Build the Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes
])

# Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# Save the Model in Recommended Format
model_path = "MachineLearning/model/chickenpox_model_improved.keras"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"Model saved to {model_path}")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_path = "MachineLearning/model/chickenpox_model_improved.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"Model converted and saved to {tflite_model_path}")

# Plot Training History
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.show()

plot_training_history(history)
