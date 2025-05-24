# Ensure required libraries are installed:
# pip install tensorflow  # or tensorflow-cpu
# Or for newer setups: pip install keras-core

# Import necessary modules from TensorFlow/Keras
try:
    # Try the modern Keras 3 path first
    import keras_core as keras
    # Check for specific application import path in Keras Core if needed
    try:
        from keras_core.applications import VGG16 # Or ResNet50, MobileNetV2, etc.
    except ImportError:
        print("Keras Core applications might be in a different submodule or need separate install.")
        print("Trying tensorflow.keras.applications...")
        # Fallback if keras_core structure differs or Applications are separate
        try:
             from tensorflow.keras.applications import VGG16
        except ImportError:
             print("Could not import Applications module.")
             exit()
    print("Using Keras Core (keras_core) potentially with TF Keras applications")
except ImportError:
    try:
        # Fallback to TensorFlow's built-in Keras
        import tensorflow as tf
        from tensorflow.keras.applications import VGG16 # Or ResNet50, MobileNetV2, etc.
        print(f"Using TensorFlow Keras (tf version: {tf.__version__})")
    except ImportError:
        print("TensorFlow/Keras not found. Please install it: pip install tensorflow")
        exit()

# --- Load a Pre-trained CNN Model (e.g., VGG16) --- 
print("\n--- Loading a Pre-trained CNN Architecture (VGG16) --- ")

# Instantiate the VGG16 model
# Arguments:
#   include_top=True: Include the final fully-connected layers (classifier).
#                     Set to False if you only want the feature extraction part.
#   weights='imagenet': Load weights pre-trained on the ImageNet dataset.
#                      Set to None if you only want the architecture without weights.
#   input_shape=(224, 224, 3): Optional override for input shape (VGG16 default).

try:
    # We load without weights just to show the architecture quickly
    # Loading with 'imagenet' weights might trigger a download if not cached.
    model_vgg16 = VGG16(include_top=True, weights=None)
    print("\n--- VGG16 Model Architecture Loaded Successfully --- ")

    # --- Print Model Summary --- 
    # This shows the layers, output shapes, and number of parameters
    print("\n--- VGG16 Model Summary --- ")
    model_vgg16.summary()

except Exception as e:
    print(f"\nError loading VGG16 model: {e}")
    print("Ensure TensorFlow/Keras is correctly installed and network access is available if downloading weights.")


# --- Note on Other Architectures --- 
print("\n--- Note on Other Architectures --- ")
print("Keras Applications includes many other pre-trained models:")
print("- ResNet (ResNet50, ResNet101, ...)")
print("- MobileNet (MobileNet, MobileNetV2) - Efficient models")
print("- Inception (InceptionV3)")
print("- DenseNet (DenseNet121, ...)")
print("- EfficientNet (EfficientNetB0, ...)")
print("You can load them similarly, e.g., from tensorflow.keras.applications import ResNet50")

print("\nScript finished.") 