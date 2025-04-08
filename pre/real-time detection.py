import face_recognition
import cv2
import pickle
import numpy as np

# Configuration
THRESHOLD = 0.45
UNKNOWN_NAME = "Unknown"

def load_known_faces(filename):
    """Load known face encodings and names from file"""
    with open(filename, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

def recognize_faces(frame, known_encodings, known_names):
    """Recognize faces in a frame and return annotated frame"""
    # Convert to RGB and find faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calculate face distances
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_idx = np.argmin(face_distances)
        best_distance = face_distances[best_match_idx]
        
        # Determine name based on threshold
        if best_distance <= THRESHOLD:
            name = known_names[best_match_idx]
            color = (0, 255, 0)  # Green for known faces
        else:
            name = UNKNOWN_NAME
            color = (0, 0, 255)  # Red for unknown faces
        
        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, bottom + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def main():
    # Load known faces
    known_encodings, known_names = load_known_faces("face_encodings.pkl")
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Recognize and display faces
        frame = recognize_faces(frame, known_encodings, known_names)
        cv2.imshow('Video', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()