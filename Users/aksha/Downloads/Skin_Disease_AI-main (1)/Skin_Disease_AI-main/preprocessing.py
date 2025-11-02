"""
Data Preprocessing Script for Skin Disease AI
Loads and preprocesses skin lesion images, then splits data into train/validation/test sets.
"""
import os
import pickle
import numpy as np
import seaborn as sns
from tqdm import tqdm
import tensorflow.keras as K
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import img_to_array

# Load all images from the dataset directory
img_path = os.listdir('dataset_dir')

# Initialize lists to store features and labels
features = []
labels = []

# Class mapping dictionary
CLASS_MAPPING = {
    'acne': 0, 
    'carcinoma': 1, 
    'eczema': 2, 
    'keratosis': 3, 
    'mila': 4, 
    'rosacea': 5
}

# Process each image in the dataset
print("Loading and preprocessing images...")
for i in tqdm(img_path):
    # Construct full file path
    file_path = os.path.join('dataset_dir', i)
    
    # Load and preprocess image (299x299 for Xception model)
    img = image.load_img(file_path, target_size=(299, 299))
    img_array = img_to_array(img)
    
    # Apply Xception-specific preprocessing
    processed_img = K.applications.xception.preprocess_input(img_array)
    features.append(processed_img)
    
    # Extract label from filename (assumes filename format: classname.extension)
    class_name = i.split(".")[0]
    labels.append(CLASS_MAPPING[class_name])

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Visualize class distribution before augmentation
print("Generating class distribution visualization...")
unique_labels, label_counts = np.unique(labels, return_counts=True)

# Map numeric labels back to class names for visualization
label_to_name = {
    0: 'acne', 
    1: 'carcinoma', 
    2: 'eczema', 
    3: 'keratosis', 
    4: 'millia', 
    5: 'rosacea'
}

class_distribution = {}
for i in range(len(unique_labels)):
    class_distribution[label_to_name[unique_labels[i]]] = label_counts[i]

# Create bar plot
sns.set_theme(style="whitegrid")
ax = sns.barplot(
    x=list(class_distribution.keys()), 
    y=list(class_distribution.values())
)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container)

plt.title('Class Distribution Before Augmentation')
plt.xlabel('Skin Conditions')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Shuffle the data to ensure random distribution
print("Shuffling data...")
indices = np.random.permutation(len(features))
features = features[indices]
labels = labels[indices]

# Initialize train/validation/test sets
x_train, y_train = [], []
x_val, y_val = [], []
x_test, y_test = [], []
x_train_temp, y_train_temp = [], []  # Initialize these variables

# Split data: 80% for training/testing, 20% for final testing
print("Splitting data into train/test sets...")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in split.split(features, labels):
    # Training data (will be further split for validation)
    x_train_temp = features[train_index]
    y_train_temp = labels[train_index]
    
    # Final test data
    x_test = features[test_index]
    y_test = labels[test_index]

# Further split training data: 85% for training, 15% for validation
print("Splitting training data into train/validation sets...")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
for val_index, train_index in split.split(x_train_temp, y_train_temp):
    x_val = x_train_temp[val_index]
    y_val = y_train_temp[val_index]
    
    x_train = x_train_temp[train_index]
    y_train = y_train_temp[train_index]

# Save processed data to pickle files
print("Saving processed data...")

# Save training data
with open("processed_data/x_train", "wb") as f:
    pickle.dump(x_train, f)
    
with open("processed_data/y_train", "wb") as f:
    pickle.dump(y_train, f)

# Save validation data
with open("processed_data/x_val", "wb") as f:
    pickle.dump(x_val, f)
    
with open("processed_data/y_val", "wb") as f:
    pickle.dump(y_val, f)

# Save test data
with open("processed_data/x_test", "wb") as f:
    pickle.dump(x_test, f)
    
with open("processed_data/y_test", "wb") as f:
    pickle.dump(y_test, f)

print("Preprocessing complete! Data saved to processed_data directory.")
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")