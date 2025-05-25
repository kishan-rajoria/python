# Ensure required libraries are installed:
# pip install tensorflow  # or tensorflow-cpu numpy matplotlib
# Or for newer setups: pip install keras-core

import numpy as np
from matplotlib import pyplot as plt

# Import necessary modules from TensorFlow/Keras
try:
    # Try the modern Keras 3 path first
    import keras_core as keras
    from keras_core import layers
    try:
        from keras_core.applications import MobileNetV2 # Using MobileNetV2 as a common, smaller example
    except ImportError:
        print("Keras Core applications might be in a different submodule or need separate install.")
        print("Trying tensorflow.keras.applications...")
        try:
             from tensorflow.keras.applications import MobileNetV2
        except ImportError:
             print("Could not import Applications module.")
             exit()
    print("Using Keras Core (keras_core) potentially with TF Keras applications")
except ImportError:
    try:
        # Fallback to TensorFlow's built-in Keras
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.applications import MobileNetV2 # Using MobileNetV2 as a common, smaller example
        print(f"Using TensorFlow Keras (tf version: {tf.__version__})")
    except ImportError:
        print("TensorFlow/Keras not found. Please install it: pip install tensorflow")
        exit()

# --- Configuration --- 
input_shape = (128, 128, 3) # Example input shape - adjust based on base model needs & data
num_classes_new_task = 5  # Example: adapting to a new task with 5 classes

print("\n--- Transfer Learning Demonstration --- ")

# --- 1. Load Pre-trained Base Model --- 
print(f"\nLoading base model (MobileNetV2) with pre-trained ImageNet weights...")
print("  include_top=False: Excludes the original classification layer.")

base_model = None
try:
    base_model = MobileNetV2(
        weights='imagenet',      # Load weights pre-trained on ImageNet
        include_top=False,     # Exclude the final classification layer
        input_shape=input_shape # Specify input shape for the base model
    )
    print("Base model loaded successfully.")
except Exception as e:
    print(f"Error loading base model: {e}")
    print("Ensure network access if weights need downloading.")
    exit()

# --- 2. Freeze the Base Model --- 
print("\nFreezing weights of the base model...")
# This prevents the weights in the pre-trained layers from being updated during initial training
base_model.trainable = False
print(f"Base model trainable status: {base_model.trainable}")

# --- 3. Add Custom Layers on Top --- 
print("\nAdding custom classification layers...")

# Start building the new model sequentially
inputs = keras.Input(shape=input_shape)
# Pass inputs through the (frozen) base model
x = base_model(inputs, training=False) # Important: Set training=False for frozen layers

# Add layers suitable for classification on top of the base model's features
# GlobalAveragePooling2D reduces spatial dimensions to a single vector per feature map
x = layers.GlobalAveragePooling2D()(x)
# Optional: Add a Dense layer for further processing
x = layers.Dense(128, activation='relu')(x)
# Optional: Add Dropout for regularization
x = layers.Dropout(0.3)(x)
# Final classification layer for the new task
outputs = layers.Dense(num_classes_new_task, activation='softmax')(x)

# --- 4. Create the Final Model --- 
model_transfer = keras.Model(inputs, outputs)
print("\nFinal transfer learning model created.")

# --- 5. Print Model Summary --- 
print("\n--- Transfer Learning Model Summary --- ")
model_transfer.summary()

print("\nObserve the large number of non-trainable parameters (from frozen base model)")
print("and the smaller number of trainable parameters (from the new top layers).")

# --- Note on Compilation and Training --- 
print("\n--- Note on Compilation and Training --- ")
print("Next steps would be:")
print("1. Compile the model:")
print(f"   model_transfer.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])")
print("2. Train the model on your new dataset (only the top layers train initially):")
print("   model_transfer.fit(new_dataset_train, epochs=..., validation_data=new_dataset_val)")

# --- 6. Note on Fine-tuning (Optional - after initial training) --- 
print("\n--- Note on Fine-tuning (Optional Step) --- ")
print("After the new layers have converged, you can optionally unfreeze some layers")
print("of the base model and continue training with a very low learning rate.")
print("Example (commented out):")
# print("\nUnfreezing some base model layers for fine-tuning...")
# base_model.trainable = True # Unfreeze the whole base
# # Or selectively unfreeze layers, e.g., layers after a certain index
# fine_tune_at = 100 # Example: Unfreeze layers from index 100 onwards
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False
#
# print("Re-compile the model with a low learning rate:")
# model_transfer.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Very low LR
#                        loss='categorical_crossentropy',
#                        metrics=['accuracy'])
# print("Continue training (fine-tuning):")
# model_transfer.fit(new_dataset_train, epochs=..., validation_data=new_dataset_val) # Train for a few more epochs

print("\nScript finished.") 