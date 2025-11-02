"""
Model Evaluation Script for Skin Disease AI
Evaluates the trained model performance using test data.
Generates confusion matrix, accuracy/loss curves, and classification reports.
"""
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import (
    ConfusionMatrixDisplay, 
    confusion_matrix, 
    classification_report
)
import matplotlib.pyplot as plt

# Load test data
print("Loading test data...")
x_test = pickle.load(open('processed_data/x_test', 'rb'))
y_test = pickle.load(open('processed_data/y_test', 'rb'))

# Load training history
print("Loading training history...")
with open('training_history/phase2_history', "rb") as file_pi:
    training_history = pickle.load(file_pi)

# Load trained model
print("Loading trained model...")
model = tf.keras.models.load_model("final_trained_model")

# Class names for skin conditions
CLASS_NAMES = ['acne', 'carcinoma', 'eczema', 'keratosis', 'mila', 'rosacea']

# Generate predictions on test set
print("Generating predictions on test set...")
predictions = model.predict(x_test)
test_predictions = np.argmax(predictions, axis=1)

# Generate and display confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, test_predictions)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=CLASS_NAMES
)
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Skin Disease Classification')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot training and validation accuracy
print("Plotting accuracy curves...")
train_acc = [0.0] + training_history['accuracy']
val_acc = [0.0] + training_history['val_accuracy']

plt.figure(figsize=(12, 6))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Model Accuracy Over Training Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.grid(True)
plt.show()

# Plot training and validation loss
print("Plotting loss curves...")
train_loss = [0.0] + training_history['loss']
val_loss = [0.0] + training_history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss Over Training Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.grid(True)
plt.show()

# Calculate per-class accuracy
print("Per-Class Accuracy:")
for i in range(len(CLASS_NAMES)):
    class_accuracy = (cm[i][i] / sum(cm[i])) * 100
    print(f'{CLASS_NAMES[i].capitalize()}: {class_accuracy:.2f}%')

# Generate detailed classification report
print("\nDetailed Classification Report:")
report = classification_report(
    y_test, 
    test_predictions, 
    target_names=[name.capitalize() for name in CLASS_NAMES]
)
print(report)

# Calculate overall test accuracy
overall_accuracy = np.sum(test_predictions == y_test) / len(y_test) * 100
print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")

print("Evaluation complete!")