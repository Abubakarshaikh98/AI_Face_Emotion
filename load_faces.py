import cv2
import os

# =============================
# Models
# =============================
detector_model = "models/face_detection_yunet_2026may.onnx"
recognizer_model = "models/face_recognition_sface_2021dec.onnx"

# Face Detector
detector = cv2.FaceDetectorYN.create(
    detector_model,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

# Face Recognizer
recognizer = cv2.FaceRecognizerSF.create(
    recognizer_model,
    ""
)

# =============================
# Known faces folder
# =============================
known_faces_folder = "known_faces"

# Store embeddings
known_features = {}

# =============================
# Load every person's folder
# =============================
for person_name in os.listdir(known_faces_folder):

    person_folder = os.path.join(
        known_faces_folder,
        person_name
    )

    # Skip files
    if not os.path.isdir(person_folder):
        continue

    # Check images
    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        # =============================
        # Detect face
        # =============================
        height, width = image.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        _, faces = detector.detect(image)

        if faces is None:
            print(
                f"No face found: {image_path}"
            )
            continue

        # =============================
        # Take first face
        # =============================
        face = faces[0]

        # =============================
        # Align face
        # =============================
        aligned_face = recognizer.alignCrop(
            image,
            face
        )

        # =============================
        # Create embedding
        # =============================
        feature = recognizer.feature(
            aligned_face
        )

        # =============================
        # Save embedding
        # =============================
        known_features[person_name] = feature

        print(
            f"Loaded: {person_name}"
        )

        break


# =============================
# Show loaded people
# =============================
print("\nKnown people:")

for name in known_features:
    print("-", name)

print(
    f"\nTotal people: {len(known_features)}"
)