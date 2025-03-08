import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

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
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the Euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize webcam
video_capture = cv2.VideoCapture(0)

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

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()