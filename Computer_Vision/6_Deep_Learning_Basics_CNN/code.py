# Ensure required libraries are installed:
# pip install tensorflow  # or tensorflow-cpu
# Or for newer setups: pip install keras-core

# Import necessary modules from TensorFlow/Keras
try:
    # Try the modern Keras 3 path first
    import keras_core as keras
    from keras_core import layers
    print("Using Keras Core (keras_core)")
except ImportError:
    try:
        # Fallback to TensorFlow's built-in Keras
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        print(f"Using TensorFlow Keras (tf version: {tf.__version__})")
    except ImportError:
        print("TensorFlow/Keras not found. Please install it: pip install tensorflow")
        exit()

# --- Define a Simple CNN Model --- 
print("\n--- Defining a Basic Convolutional Neural Network (CNN) --- ")

# Define input shape (example: 32x32 color image)
# In a real scenario, this depends on your dataset
input_shape = (32, 32, 3) # height, width, channels (3 for RGB)
num_classes = 10 # Example: 10 classes for digit recognition or simple object categories

# Build the Sequential model
model = keras.Sequential(
    [
        # Input Layer - specifying the input shape is crucial for the first layer
        keras.Input(shape=input_shape),

        # --- Convolutional Block 1 ---
        # Conv2D: Applies convolution operation (learns spatial hierarchies)
        #   32 filters: Number of output filters (feature maps)
        #   kernel_size=(3, 3): Size of the convolution window
        #   activation='relu': Rectified Linear Unit activation function (introduces non-linearity)
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),

        # MaxPooling2D: Downsamples the feature maps (reduces dimensionality, provides invariance)
        #   pool_size=(2, 2): Factor by which to downscale (2x2 window)
        layers.MaxPooling2D(pool_size=(2, 2)),

        # --- Convolutional Block 2 ---
        # Another Conv layer, often with more filters
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # Another Max Pooling layer
        layers.MaxPooling2D(pool_size=(2, 2)),

        # --- Flattening ---
        # Flatten: Converts the 2D feature maps into a 1D vector
        # This prepares the output of convolutional layers for the Dense layers
        layers.Flatten(),

        # --- Optional: Dropout Layer ---
        # Dropout: Regularization technique to prevent overfitting
        #   0.5: Fraction of the input units to drop during training
        # layers.Dropout(0.5),

        # --- Dense (Fully Connected) Layer ---
        # Dense: Standard fully connected layer
        #   num_classes: Number of output units, corresponding to the number of classes
        #   activation='softmax': Converts output scores into probabilities (for multi-class classification)
        layers.Dense(num_classes, activation="softmax"),
    ]
)

print("\n--- CNN Model Defined Successfully --- ")

# --- Print Model Summary --- 
# This shows the layers, output shapes, and number of parameters
print("\n--- Model Summary --- ")
model.summary()

# --- Note on Training --- 
print("\n--- Note on Training --- ")
print("This script only DEFINES the CNN structure.")
print("Actual training requires:")
print("1. Loading and preprocessing a dataset (e.g., CIFAR-10, MNIST).")
print("2. Compiling the model (specifying optimizer, loss function, metrics):")
print("   e.g., model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])")
print("3. Training the model on the data (fitting):")
print("   e.g., model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)")

print("\nScript finished.") 