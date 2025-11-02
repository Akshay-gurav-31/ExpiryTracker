"""
Skin Disease Classification Model
Builds and trains an Xception-based CNN model for skin disease diagnosis.
Uses transfer learning with ImageNet pre-trained weights.
"""
import pickle
from tensorflow.keras import Model
import tensorflow.keras as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout

# Load preprocessed and augmented training data
print("Loading training data...")
x_train = pickle.load(open('processed_data_after_aug/x_train', 'rb'))
y_train = pickle.load(open('processed_data_after_aug/y_train', 'rb'))

# Load validation data
x_val = pickle.load(open('processed_data_after_aug/x_val', 'rb'))
y_val = pickle.load(open('processed_data_after_aug/y_val', 'rb'))

# Load test data
x_test = pickle.load(open('processed_data_after_aug/x_test', 'rb'))
y_test = pickle.load(open('processed_data_after_aug/y_test', 'rb'))

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")

# Build the model using Xception architecture with transfer learning
print("Building model with Xception base...")

# Load Xception base model with ImageNet weights
base_model = K.applications.Xception(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(299, 299, 3),
    pooling=None,
    classifier_activation="softmax",
)

# Freeze the base model for initial training
base_model.trainable = False

# Define model inputs
inputs = K.Input(shape=(299, 299, 3))

# Connect base model to inputs
x = base_model(inputs, training=False)

# Add custom classification layers
x = GlobalAveragePooling2D()(x)  # Reduce spatial dimensions
x = BatchNormalization()(x)      # Normalize inputs for stable training
x = Dropout(0.3)(x)              # Prevent overfitting

# First dense layer
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Second dense layer
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Output layer (6 classes for skin conditions)
outputs = Dense(6, activation='softmax')(x)

# Create the complete model
model = Model(inputs, outputs)

# Compile model for initial training phase
print("Compiling model for initial training...")
optimizer = Adam(learning_rate=0.001)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Set up model checkpoints to save best models
acc_checkpoint = ModelCheckpoint(
    "training_checkpoints/best_accuracy_model", 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

loss_checkpoint = ModelCheckpoint(
    "training_checkpoints/best_loss_model", 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

callbacks_list = [acc_checkpoint, loss_checkpoint]

# Phase 1: Train custom layers only (transfer learning)
print("Starting Phase 1 training (15 epochs)...")
history_phase1 = model.fit(
    x_train, 
    y_train, 
    epochs=15, 
    validation_data=(x_val, y_val), 
    batch_size=32, 
    callbacks=callbacks_list
)

# Save training history
with open('training_history/phase1_history', 'wb') as file_pi:
    pickle.dump(history_phase1.history, file_pi)

# Phase 2: Fine-tune entire model
print("Starting Phase 2 training (fine-tuning)...")

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Recompile with lower learning rate for fine-tuning
optimizer = Adam(learning_rate=0.00001)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Update checkpoint paths for phase 2
acc_checkpoint = ModelCheckpoint(
    "training_checkpoints/phase2_best_accuracy_model", 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

loss_checkpoint = ModelCheckpoint(
    "training_checkpoints/phase2_best_loss_model", 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

callbacks_list = [acc_checkpoint, loss_checkpoint]

# Train for additional 10 epochs with fine-tuning
history_phase2 = model.fit(
    x_train, 
    y_train, 
    epochs=10, 
    validation_data=(x_val, y_val), 
    batch_size=32, 
    callbacks=callbacks_list
)

# Save phase 2 training history
with open('training_history/phase2_history', 'wb') as file_pi:
    pickle.dump(history_phase2.history, file_pi)

# Save the final trained model
print("Saving final model...")
model.save('final_trained_model')

print("Model training complete!")
print(f"Final validation accuracy: {max(history_phase2.history['val_accuracy']):.4f}")