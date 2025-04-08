import cv2
import pickle
import numpy as np
from face_recognition import face_encodings, face_distance

# Load the stored encodings
def load_encodings():
    try:
        with open('data/encodings.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data['encodings'])} encodings for {len(set(data['names']))} people")
        return data['encodings'], data['names']
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []

known_encodings, known_names = load_encodings()

if not known_encodings:
    print("No encodings found! Please run face collection first.")
    exit()

# Initialize video capture
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open video capture")
    exit()

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if facedetect.empty():
    print("Error: Could not load face detector")
    exit()

# Recognition parameters
TOLERANCE = 0.45  # Lower is more strict (typical range 0.5-0.6)
FRAME_SKIP = 1  # Process every 3rd frame for performance

frame_count = 0
recognition_debug = []

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Couldn't read frame")
        break
    
    frame_count += 1
    
    # Only process every FRAME_SKIP-th frame to improve performance
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        recognition_debug.append("No faces detected")
    
    for (x, y, w, h) in faces:
        # Extract the face ROI and convert to RGB
        face_roi = frame[y:y+h, x:x+w]
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        try:
            # Get 128D encoding for the face
            face_encoding = face_encodings(face_roi_rgb)
            
            if len(face_encoding) == 0:
                recognition_debug.append("No face encoding generated")
                continue
                
            face_encoding = face_encoding[0]
            
            # Compare with known encodings
            distances = face_distance(known_encodings, face_encoding)
            best_match_idx = np.argmin(distances)
            min_distance = distances[best_match_idx]
            
            if min_distance <= TOLERANCE:
                name = known_names[best_match_idx]
                confidence = 1 - min_distance
                color = (0, 255, 0)  # Green for recognized
                text = f"{name} ({confidence:.2f})"
                recognition_debug.append(f"Recognized {name} with confidence {confidence:.2f}")
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
                text = name
                recognition_debug.append(f"Unknown face (min distance: {min_distance:.2f})")
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, text, (x+6, y-6), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        except Exception as e:
            recognition_debug.append(f"Error processing face: {str(e)}")
            continue
    
    # Display debug info (last 3 messages)
    for i, msg in enumerate(recognition_debug[-3:]):
        cv2.putText(frame, msg, (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()