import os
import cv2
import h5py
import numpy as np
import face_recognition

# Define dataset path
DATASET_FOLDER = r"C:\Users\potta\Desktop\dataset\train"
HDF5_FILE = "face_embeddings.h5"

# Function to detect faces and extract embeddings
def process_image(image_path):
    """Detect face and extract embedding from an image."""
    try:
        # Load image
        image = face_recognition.load_image_file(image_path)

        # Detect face locations
        face_locations = face_recognition.face_locations(image, model="hog")

        # If no face is detected, return None
        if not face_locations:
            print(f"⚠️ No face detected in {image_path}, skipping...")
            return None, None

        # Extract facial encoding (only the first detected face)
        face_encodings = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])

        if not face_encodings:
            print(f"⚠️ No face encoding found in {image_path}, skipping...")
            return None, None

        return face_encodings[0], image_path  # Return the first detected face encoding

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None, None
# Step 1: Generate and Save Embeddings
dataset_embeddings = []
dataset_image_paths = []

print("🔍 Processing images...")

for img_name in os.listdir(DATASET_FOLDER):
    img_path = os.path.join(DATASET_FOLDER, img_name)
    
    # Process each image
    embedding, path = process_image(img_path)
    if embedding is not None:
        dataset_embeddings.append(embedding)
        dataset_image_paths.append(path)

# Step 2: Save Embeddings to HDF5
if dataset_embeddings:
    dataset_embeddings = np.array(dataset_embeddings, dtype=np.float32)

    with h5py.File(HDF5_FILE, "w") as f:
        f.create_dataset("embeddings", data=dataset_embeddings)
        f.create_dataset("image_paths", data=np.array(dataset_image_paths, dtype="S"))

    print("✅ Embeddings saved successfully.")
else:
    print("❌ No embeddings were generated.")