import face_recognition
import cv2
import pickle
import csv
from datetime import datetime
import os


# Load the saved face encodings
with open("face_encodings.pkl", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Define the CSV file path
attendance_file = "attendance.csv"

# Create the CSV file with headers if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# Track recognized faces to avoid duplicate entries for the same day
recognized_faces_today = set()

def is_attendance_marked_today(name):
    """
    Check if attendance for the given name has already been marked today.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    with open(attendance_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == name and row[1].startswith(today):
                return True
    return False

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Mark attendance if the face is recognized and not already logged today
            if name not in recognized_faces_today and not is_attendance_marked_today(name):
                recognized_faces_today.add(name)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, timestamp])
                print(f"Attendance marked for {name} at {timestamp}")

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw the name below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()