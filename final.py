import cv2
import dlib
import numpy as np
import pickle
import os
import csv
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils
import face_recognition

# Constants for eye blink detection
EYE_AR_THRESH = 0.25  # Eye Aspect Ratio threshold
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames to confirm a blink

# Initialize counters and flags
COUNTER = 0
TOTAL = 0
BLINK_DETECTED = False

# Load the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to encode faces from the dataset
def encode_faces(dataset_folder):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_folder):
        person_folder = os.path.join(dataset_folder, person_name)
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = face_recognition.load_image_file(image_path)

            # Detect faces in the image
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) > 0:
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)

    # Save the encodings to a pickle file
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print("Face encodings saved to face_encodings.pkl")

# Load the saved face encodings
def load_face_encodings():
    with open("face_encodings.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Create or open the attendance log file
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# Track recognized faces to avoid duplicate entries
recognized_faces = set()

# Encode faces from the dataset (run this only once)
# encode_faces("dataset")

# Load the saved face encodings
known_face_encodings, known_face_names = load_face_encodings()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Loop over the face detections
    for face in faces:
        # Detect facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate the EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Check if the EAR is below the blink threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            # If the eyes were closed for enough frames, increment the blink counter
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                BLINK_DETECTED = True
            COUNTER = 0

        # Draw the eye contours on the frame
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        # Display the EAR and blink count
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Perform face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # Compare the face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Mark attendance if the face is recognized and a blink is detected
                if name not in recognized_faces and BLINK_DETECTED:
                    recognized_faces.add(name)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(attendance_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, timestamp])
                    print(f"Attendance marked for {name} at {timestamp}")
                    BLINK_DETECTED = False  # Reset the flag

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()