import os
import h5py
import numpy as np
import cv2
import face_recognition
import faiss
from flask import Flask, render_template, request, jsonify, send_from_directory
import base64

app = Flask(__name__, template_folder="templates")

# Paths
UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = r"C:\Users\potta\Desktop\dataset\train"  # Update based on your dataset location
HDF5_FILE = "face_embeddings.h5"
THRESHOLD = 0.3  # Reduced threshold for better accuracy
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# **Load Saved Embeddings**
print("🔄 Loading saved embeddings...")
if os.path.exists(HDF5_FILE):
    with h5py.File(HDF5_FILE, "r") as f:
        dataset_embeddings = f["embeddings"][:]
        dataset_image_paths = [path.decode() for path in f["image_paths"][:]]

    if dataset_embeddings.shape[0] > 0:
        dataset_embeddings /= np.linalg.norm(dataset_embeddings, axis=1, keepdims=True)
        index = faiss.IndexFlatL2(dataset_embeddings.shape[1])
        index.add(dataset_embeddings)
    else:
        dataset_embeddings = None
else:
    dataset_embeddings = None

# **Face Alignment & Preprocessing**
def align_face(image):
    """Detects and aligns face before extracting embeddings."""
    face_landmarks = face_recognition.face_landmarks(image)
    if not face_landmarks:
        return None  # No face detected

    # Get eye positions for alignment
    left_eye = np.mean(face_landmarks[0]['left_eye'], axis=0)
    right_eye = np.mean(face_landmarks[0]['right_eye'], axis=0)

    # Compute angle for rotation
    dx, dy = right_eye - left_eye
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate image to align eyes
    center = tuple(np.mean([left_eye, right_eye], axis=0))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    aligned_image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

    return aligned_image

@app.route("/")
def home():
    """Serve the HTML page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload, extract face embeddings, and match against the database."""
    data = request.json.get("image")
    if not data:
        return jsonify({"message": "No image data received"}), 400

    # Convert base64 to image
    image_data = base64.b64decode(data.split(",")[1])
    file_path = os.path.join(UPLOAD_FOLDER, "captured_image.png")

    with open(file_path, "wb") as f:
        f.write(image_data)

    # Ensure file is saved
    if not os.path.exists(file_path):
        return jsonify({"message": "Error saving image"}), 500

    # **Perform Face Recognition**
    test_image = face_recognition.load_image_file(file_path)
    test_image = align_face(test_image)  # Align face before encoding

    if test_image is None:
        return jsonify({"message": "No face detected in the image"}), 400

    test_face_encodings = face_recognition.face_encodings(test_image)
    if not test_face_encodings:
        return jsonify({"message": "No face encoding found in the image"}), 400

    test_embedding = np.array(test_face_encodings[0], dtype=np.float32)
    test_embedding /= np.linalg.norm(test_embedding)  # Normalize

    # **Perform similarity search**
    if dataset_embeddings is None:
        return jsonify({"message": "No stored embeddings found"}), 400

    test_embedding = test_embedding.reshape(1, -1)
    k = min(len(dataset_embeddings), 5)  # Retrieve top 5 matches
    distances, indices = index.search(test_embedding, k=k)

    # **Find valid matches**
    matching_images = []
    for i, dist in zip(indices[0], distances[0]):
        if 0 <= i < len(dataset_image_paths) and dist < THRESHOLD:
            # Extract filename from the full local path
            image_filename = os.path.basename(dataset_image_paths[i])
            image_url = f"/dataset/{image_filename}"  # Create a URL to access the image

            matching_images.append({"image": image_url, "distance": round(float(dist), 2)})

    # Debugging: Print the content of matching_images
    print("🔍 Debug: matching_images =", matching_images)

    return jsonify({
        "message": "Image processed successfully",
        "matches": matching_images,
        "uploaded_image": f"/uploads/captured_image.png"  # Ensure correct path format
    })

# **Serve Uploaded Images**
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve images from the uploads folder."""
    return send_from_directory(UPLOAD_FOLDER, filename)

# **Serve Dataset Images**
@app.route('/dataset/<filename>')
def dataset_file(filename):
    """Serve dataset images from the dataset directory."""
    return send_from_directory(DATASET_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
