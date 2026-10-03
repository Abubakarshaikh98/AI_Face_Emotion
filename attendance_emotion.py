import cv2
import csv
import os
import numpy as np
from datetime import datetime


# =============================
# Model paths
# =============================
detector_model = "models/face_detection_yunet_2026may.onnx"
recognizer_model = "models/face_recognition_sface_2021dec.onnx"
emotion_model_path = "models/facial_expression_recognition_mobilefacenet_2022july.onnx"

attendance_file = "attendance/attendance_emotion.csv"


# =============================
# Load models
# =============================
detector = cv2.FaceDetectorYN.create(
    detector_model,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    recognizer_model,
    ""
)

emotion_model = cv2.dnn.readNetFromONNX(
    emotion_model_path
)


# =============================
# Emotion labels
# =============================
emotions = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


# =============================
# Load known faces
# =============================

known_faces_folder = "known_faces"

known_features = {}


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


        # Detect face
        height, width = image.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        _, faces = detector.detect(
            image
        )


        if faces is None:
            print(
                f"No face found: {image_path}"
            )
            continue


        # First detected face
        known_face = faces[0]


        # Align face
        known_aligned = recognizer.alignCrop(
            image,
            known_face
        )


        # Create embedding
        known_feature = recognizer.feature(
            known_aligned
        )


        # Save embedding
        known_features[person_name] = known_feature


        print(
            f"Known face loaded: {person_name}"
        )

        break


# Show loaded people
print("\nKnown people:")

for name in known_features:
    print("-", name)


print(
    f"\nTotal people: {len(known_features)}"
)

# =============================
# Create attendance folder
# =============================
os.makedirs(
    "attendance",
    exist_ok=True
)


# =============================
# Create CSV
# =============================
if not os.path.exists(attendance_file):

    with open(
        attendance_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Name",
            "Date",
            "Time",
            "Emotion"
        ])


# =============================
# Mark attendance
# =============================
def mark_attendance(name, emotion):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )


    with open(
        attendance_file,
        "r",
        newline=""
    ) as file:

        reader = csv.reader(file)

        rows = list(reader)


    # Check duplicate
    for row in rows[1:]:

        if len(row) >= 2:

            existing_name = row[0]
            existing_date = row[1]

            if (
                existing_name == name
                and existing_date == today
            ):
                return False


    # Save attendance
    with open(
        attendance_file,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            name,
            today,
            current_time,
            emotion
        ])

    return True


# =============================
# Open webcam
# =============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()


print("Attendance + Emotion system started!")
print("Press Q to quit.")


# =============================
# Webcam loop
# =============================
while True:

    ret, frame = cap.read()

    if not ret:
        break


    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )


    # Detect faces
    _, faces = detector.detect(frame)


    if faces is not None:

        for face in faces:

            # -------------------------
            # Coordinates
            # -------------------------
            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)


            # =========================
            # FACE RECOGNITION
            # =========================

            aligned_face = recognizer.alignCrop(
                frame,
                face
            )

            feature = recognizer.feature(
                aligned_face
            )

                        # =========================
            # Compare with all known faces
            # =========================

            best_name = "Unknown"
            best_score = 0.0

            for person_name, known_feature in known_features.items():

                score = recognizer.match(
                    known_feature,
                    feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:
                    best_score = score
                    best_name = person_name


            # =========================
            # Recognition threshold
            # =========================

            if best_score >= 0.363:

                name = best_name
                color = (0, 255, 0)

            else:

                name = "Unknown"
                color = (0, 0, 255)

           
            # =========================
            # EMOTION DETECTION
            # =========================

            emotion = "Unknown"

            face_img = frame[
                y1:y2,
                x1:x2
            ]


            if face_img.size != 0:

                face_img = cv2.resize(
                    face_img,
                    (112, 112)
                )

                blob = cv2.dnn.blobFromImage(
                    face_img,
                    scalefactor=1 / 255.0,
                    size=(112, 112),
                    mean=(0, 0, 0),
                    swapRB=False,
                    crop=False
                )

                emotion_model.setInput(blob)

                output = emotion_model.forward()

                emotion_index = np.argmax(output)

                emotion = emotions[
                    emotion_index
                ]


            # =========================
            # Save attendance
            # =========================

            if name != "Unknown":

                marked = mark_attendance(
                    name,
                    emotion
                )

                if marked:

                    print(
                        f"Attendance marked: "
                        f"{name} | {emotion}"
                    )


            # =========================
            # Draw box
            # =========================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            # =========================
            # Display
            # =========================

            text = f"{name} | {emotion}"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


    # Show webcam
    cv2.imshow(
        "AI Attendance + Emotion",
        frame
    )


    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =============================
# Cleanup
# =============================
cap.release()
cv2.destroyAllWindows()