import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

custom_objects = {
    "preprocess_input":
        tf.keras.applications.mobilenet_v2.preprocess_input
}

# Load Eye-State Model
print("Loading Eye-State Model...")

eye_model = tf.keras.models.load_model(
    MODEL_DIR / "eye_mobilenetv2_final.keras",
    custom_objects=custom_objects,
    compile=False
)

print("Eye-State Model loaded successfully!")


# Load Yawning Detection Model
print("\nLoading Yawning Detection Model...")

yawn_model = tf.keras.models.load_model(
    MODEL_DIR / "yawn_mobilenetv2_final.keras",
    custom_objects=custom_objects,
    compile=False
)

print("Yawning Detection Model loaded successfully!")


# Display Model Information
print("\n========== MODEL INFORMATION ==========")

print("Eye Model Input:", eye_model.input_shape)
print("Eye Model Output:", eye_model.output_shape)

print("Yawn Model Input:", yawn_model.input_shape)
print("Yawn Model Output:", yawn_model.output_shape)

print("\nBoth trained models loaded successfully!")


# Start Driver Monitoring System
if __name__ == "__main__":

    print("\nStarting Driver Monitoring System...")

    from app.webcam import start_webcam

    start_webcam(eye_model, yawn_model)