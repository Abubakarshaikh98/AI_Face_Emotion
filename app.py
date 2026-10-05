from flask import Flask, render_template, Response, request, session, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
import csv
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

def pakistan_now():
    return datetime.now(ZoneInfo("Asia/Karachi"))

app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY", "change-me-before-deploying")

# =========================================================
# CURRENT RECOGNITION RESULT
# =========================================================

current_result = {
    "name": "Unknown",
    "emotion": "Unknown",
    "status": "Waiting"
}


# =========================================================
# ATTENDANCE FILE
# =========================================================

attendance_file = "attendance/attendance_emotion.csv"

# (name, date) pairs already saved this run
marked_cache = set()


# =========================================================
# MARK ATTENDANCE
# =========================================================

def mark_attendance(name, emotion):

    os.makedirs("attendance", exist_ok=True)

    today = pakistan_now().strftime("%Y-%m-%d")

    if (name, today) in marked_cache:
        return

    # Check if person already marked today
    if os.path.exists(attendance_file):

        with open(attendance_file, "r", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:

                if (
                    row["Name"] == name
                    and row["Date"] == today
                ):
                    marked_cache.add((name, today))
                    return

    # Save new attendance
    file_exists = os.path.exists(attendance_file)

    with open(
        attendance_file,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "Name",
                "Date",
                "Time",
                "Emotion"
            ])

        writer.writerow([
            name,
            today,
            pakistan_now().strftime("%H:%M:%S"),
            emotion
        ])

    marked_cache.add((name, today))
    print(f"Attendance marked: {name} | {emotion}")


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)


# =========================================================
# FACE DETECTION MODEL - YUNET
# =========================================================

detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2026may.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)


# =========================================================
# FACE RECOGNITION MODEL - SFACE
# =========================================================

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)


# =========================================================
# EMOTION MODEL
# =========================================================

emotion_net = cv2.dnn.readNet(
    "models/facial_expression_recognition_mobilefacenet_2022july.onnx"
)


emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


# =========================================================
# LOAD KNOWN FACES
# =========================================================

known_features = {}

known_faces_path = "known_faces"


def load_known_faces():

    global known_features

    known_features = {}

    if not os.path.exists(known_faces_path):
        os.makedirs(
            known_faces_path,
            exist_ok=True
        )

    for person_name in os.listdir(
        known_faces_path
    ):

        person_folder = os.path.join(
            known_faces_path,
            person_name
        )

        if not os.path.isdir(
            person_folder
        ):
            continue

        for image_name in os.listdir(
            person_folder
        ):

            image_path = os.path.join(
                person_folder,
                image_name
            )

            image = cv2.imread(
                image_path
            )

            if image is None:
                continue

            detector.setInputSize(
                (
                    image.shape[1],
                    image.shape[0]
                )
            )

            _, faces = detector.detect(
                image
            )

            if faces is None:
                continue

            face = faces[0]

            aligned_face = (
                recognizer.alignCrop(
                    image,
                    face
                )
            )

            feature = recognizer.feature(
                aligned_face
            )

            known_features[
                person_name
            ] = feature

            break


    print(
        "Known faces loaded:",
        list(known_features.keys())
    )


# =========================================================
# REGISTERED PEOPLE (names of folders inside known_faces)
# =========================================================

def get_registered_names():

    names = set()

    if os.path.exists(known_faces_path):

        for person_name in os.listdir(known_faces_path):

            if os.path.isdir(
                os.path.join(known_faces_path, person_name)
            ):
                names.add(person_name)

    return names


# Load faces when app starts
load_known_faces()

# =========================================================
# GENERATE CAMERA FRAMES
# =========================================================

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        height, width = frame.shape[:2]

        # Update detector input size
        detector.setInputSize(
            (width, height)
        )

        # Detect faces
        _, faces = detector.detect(frame)


        # =================================================
        # PROCESS ALL DETECTED FACES
        # =================================================

        if faces is not None:

            for face in faces:

                # Face coordinates
                x, y, w, h = face[:4].astype(int)


                # =============================================
                # FACE RECOGNITION
                # =============================================

                aligned_face = recognizer.alignCrop(
                    frame,
                    face
                )

                feature = recognizer.feature(
                    aligned_face
                )


                name = "Unknown"

                best_score = 0


                # Compare with known faces
                for person_name, known_feature in known_features.items():

                    score = recognizer.match(
                        feature,
                        known_feature,
                        cv2.FaceRecognizerSF_FR_COSINE
                    )

                    if score > best_score:

                        best_score = score

                        name = person_name


                # Recognition threshold
                if best_score < 0.363:

                    name = "Unknown"


                # =============================================
                # FACE CROP
                # =============================================

                face_crop = frame[
                    max(0, y):min(height, y + h),
                    max(0, x):min(width, x + w)
                ]


                # =============================================
                # EMOTION DETECTION
                # =============================================

                emotion = "Unknown"


                if face_crop.size > 0:

                    emotion_input = cv2.resize(
                        face_crop,
                        (112, 112)
                    )


                    blob = cv2.dnn.blobFromImage(
                        emotion_input,
                        scalefactor=1.0 / 255.0,
                        size=(112, 112),
                        mean=(0, 0, 0),
                        swapRB=True,
                        crop=False
                    )


                    emotion_net.setInput(blob)

                    predictions = emotion_net.forward()

                    emotion_index = np.argmax(
                        predictions[0]
                    )


                    emotion = emotion_labels[
                        emotion_index
                    ]


                # =============================================
                # UPDATE FRONTEND RESULT
                # =============================================

                current_result["name"] = name

                current_result["emotion"] = emotion


                if name != "Unknown":

                    current_result["status"] = "Present"

                else:

                    current_result["status"] = "Waiting"


                # =============================================
                # MARK ATTENDANCE
                # =============================================

                if name != "Unknown":

                    mark_attendance(
                        name,
                        emotion
                    )


                # =============================================
                # DRAW FACE BOX
                # =============================================

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )


                # =============================================
                # DRAW NAME + SCORE
                # =============================================

                cv2.putText(
                    frame,
                    f"{name} ({best_score:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                # =============================================
                # DRAW EMOTION
                # =============================================

                cv2.putText(
                    frame,
                    f"Emotion: {emotion}",
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )


        # =================================================
        # CONVERT FRAME TO JPEG
        # =================================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue


        frame_bytes = buffer.tobytes()


        # Send frame to browser
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# =========================================================
# REQUIRE LOGIN FOR EVERYTHING EXCEPT LOGIN + STATIC
# =========================================================

@app.before_request
def require_login():

    if request.endpoint in ("login", "static"):
        return

    if not session.get("logged_in"):

        # API calls get 401, pages get redirected
        if request.path in ("/", "/logout"):
            return redirect(url_for("login"))

        return {"success": False, "message": "Login required."}, 401

# =========================================================
# ADMIN LOGIN
# =========================================================

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        if (
            username == ADMIN_USERNAME
            and password == ADMIN_PASSWORD
        ):

            session["logged_in"] = True

            return redirect(
                url_for("home")
            )

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template("login.html")

# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("index.html")

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))

# =========================================================
# LIVE VIDEO
# =========================================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================================================
# RECOGNITION STATUS API
# =========================================================

@app.route("/status")
def status():

    return current_result

# =========================================================
# DASHBOARD STATS API
# =========================================================

@app.route("/stats")
def stats():
    total_people = 0

    if os.path.exists(known_faces_path):
        for person_name in os.listdir(known_faces_path):
            person_folder = os.path.join(known_faces_path, person_name)

            if os.path.isdir(person_folder):
                total_people += 1

    present_today = set()
    today = pakistan_now().strftime("%Y-%m-%d")

    if os.path.exists(attendance_file):
        with open(attendance_file, "r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row.get("Date") == today:
                    present_today.add(row.get("Name"))

    return {
        "total_people": total_people,
        "present_today": len(present_today)
    }

# =========================================================
# PEOPLE API
# =========================================================

@app.route("/people")
def people():

    people_list = []

    if os.path.exists(known_faces_path):

        for person_name in os.listdir(known_faces_path):

            person_folder = os.path.join(
                known_faces_path,
                person_name
            )

            if os.path.isdir(person_folder):

                people_list.append({
                    "name": person_name,
                    "status": "Registered"
                })

    return people_list

# =========================================================
# ADD PERSON API
# =========================================================

@app.route("/add-person", methods=["POST"])
def add_person():

    name = secure_filename(request.form.get("name", "").strip())
    image = request.files.get("image")

    # Check name
    if not name:

        return {
            "success": False,
            "message": "Please enter person name."
        }, 400


    # Check image
    if image is None or image.filename == "":

        return {
            "success": False,
            "message": "Please select an image."
        }, 400


    # Person folder
    person_folder = os.path.join(
        known_faces_path,
        name
    )


    # Check duplicate person
    if os.path.exists(person_folder):

        return {
            "success": False,
            "message": "Person already registered."
        }, 400


    # Create folder
    os.makedirs(
        person_folder,
        exist_ok=True
    )


    # Save image
    image_path = os.path.join(
        person_folder,
        "face.jpg"
    )


    image.save(image_path)

    load_known_faces()

    return {
        "success": True,
        "message": f"{name} registered successfully."
    }

# =========================================================
# DELETE PERSON API
# =========================================================

@app.route("/delete-person/<name>", methods=["DELETE"])
def delete_person(name):

    name = secure_filename(name)

    person_folder = os.path.join(
        known_faces_path,
        name
    )

    # Check person exists
    if not os.path.exists(person_folder):

        return {
            "success": False,
            "message": "Person not found."
        }, 404

    # Delete person folder and images
    shutil.rmtree(
        person_folder
    )

    # Reload known faces
    load_known_faces()

    return {
        "success": True,
        "message": f"{name} deleted successfully."
    }
# =========================================================
# PERSON DETAILS API
# =========================================================

@app.route("/person/<name>")
def person_details(name):

    total_days = 0
    present_days = 0
    emotions = []

    if os.path.exists(attendance_file):

        dates = set()

        with open(
            attendance_file,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["Name"] == name:

                    dates.add(row["Date"])

                    emotions.append(
                        row["Emotion"]
                    )

        present_days = len(dates)

    # Total recorded days
    if os.path.exists(attendance_file):

        with open(
            attendance_file,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                total_days += 1

    return {
        "name": name,
        "status": "Registered",
        "present_days": present_days,
        "emotions": emotions
    }
# =========================================================
# ATTENDANCE API - DATE FILTER
# =========================================================

@app.route("/attendance")
def attendance():

    records = []

    # Get date from URL
    selected_date = request.args.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )

    if os.path.exists(attendance_file):

        with open(
            attendance_file,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["Date"] == selected_date:

                    records.append(row)

    return records


# =========================================================
# ATTENDANCE SUMMARY API
# =========================================================

@app.route("/attendance-summary")
def attendance_summary():

    selected_date = request.args.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )

    total_people = 0

    if os.path.exists(known_faces_path):

        for person_name in os.listdir(known_faces_path):

            person_folder = os.path.join(
                known_faces_path,
                person_name
            )

            if os.path.isdir(person_folder):
                total_people += 1


    # Count only people who are still registered
    registered = get_registered_names()

    present_people = set()

    if os.path.exists(attendance_file):

        with open(
            attendance_file,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if (
                    row["Date"] == selected_date
                    and row["Name"] in registered
                ):

                    present_people.add(
                        row["Name"]
                    )


    present = len(present_people)

    absent = max(total_people - present, 0)


    if total_people > 0:

        percentage = (
            present / total_people
        ) * 100

    else:

        percentage = 0


    return {
        "total_people": total_people,
        "present": present,
        "absent": absent,
        "percentage": round(
            percentage,
            2
        )
    }

# =========================================================
# EMOTION ANALYSIS API
# =========================================================

@app.route("/emotion-summary")
def emotion_summary():

    emotion_counts = {
        "angry": 0,
        "disgust": 0,
        "fear": 0,
        "happy": 0,
        "sad": 0,
        "surprise": 0,
        "neutral": 0
    }

    if os.path.exists(attendance_file):

        with open(
            attendance_file,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                emotion = row["Emotion"].lower()

                if emotion in emotion_counts:

                    emotion_counts[emotion] += 1

    return emotion_counts

# =========================================================
# EXPORT ATTENDANCE CSV
# =========================================================

@app.route("/export-attendance")
def export_attendance():

    if not os.path.exists(attendance_file):

        return {
            "success": False,
            "message": "Attendance file not found."
        }, 404


    return Response(
        open(
            attendance_file,
            "rb"
        ),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=attendance.csv"
        }
    )
# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        use_reloader=False
    )