import face_recognition
import cv2
import pickle
import numpy as np
import os
from datetime import datetime

# Configuration
THRESHOLD = 0.4  # Similarity threshold
UNKNOWN_NAME = "Unknown"
MIN_FACE_SIZE = 100  # Minimum pixel size for a face to be considered
BLUR_THRESHOLD = 100  # Threshold for blur detection (lower is more blurry)
SCREEN_REFLECTION_THRESHOLD = 200  # Brightness threshold for screen detection

def load_known_faces(filename):
    """Load known face encodings and names from file with error handling"""
    try:
        with open(filename, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
        return known_face_encodings, known_face_names
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        print(f"Error loading face database: {e}")
        return [], []

def is_screen_face(frame, face_location):
    """Check if the face is likely on a screen (photo/video)"""
    (top, right, bottom, left) = face_location
    face_region = frame[top:bottom, left:right]
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_face)
    return avg_brightness > SCREEN_REFLECTION_THRESHOLD

def is_too_blurry(frame):
    """Check if the frame is too blurry for reliable detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

def is_face_too_small(face_location):
    """Check if the detected face is too small to be real"""
    (top, right, bottom, left) = face_location
    return (right - left) < MIN_FACE_SIZE or (bottom - top) < MIN_FACE_SIZE

def recognize_faces(frame, known_encodings, known_names):
    """Enhanced face recognition with filtering"""
    if is_too_blurry(frame):
        cv2.putText(frame, "Low quality - ignoring", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    
    for face_location in face_locations:
        if is_face_too_small(face_location) or is_screen_face(frame, face_location):
            continue
            
        face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_idx = np.argmin(face_distances)
        best_distance = face_distances[best_match_idx]
        
        name = known_names[best_match_idx] if best_distance <= THRESHOLD else UNKNOWN_NAME
        (top, right, bottom, left) = face_location
        
        # Draw rectangle and label
        color = (0, 255, 0) if name != UNKNOWN_NAME else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, bottom + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Display confidence score
        confidence = 1 - best_distance
        cv2.putText(frame, f"{confidence:.2f}", (left, bottom + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return frame

def log_attendance(name):
    """Log recognized faces with timestamp"""
    if name != UNKNOWN_NAME:
        with open("attendance.csv", "a") as f:
            f.write(f"{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    known_encodings, known_names = load_known_faces("face_encodings.pkl")
    
    if not known_encodings:
        print("No face database found. Please create one first.")
        return
    
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        frame = recognize_faces(frame, known_encodings, known_names)
        
        # # Display FPS
        # cv2.putText(frame, f"FPS: {int(video_capture.get(cv2.CAP_PROP_FPS))}", 
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('a'):  # Add new face to database
            add_new_face(frame, known_encodings, known_names)
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()