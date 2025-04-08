import cv2
import pickle
import numpy as np
import random
import time
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
TOLERANCE = 0.45
FRAME_SKIP = 1

# Liveness detection parameters
CHALLENGE_DURATION = 7  # seconds per challenge
CHALLENGES = ["turn_head", "smile", "blink", "nod"]

frame_count = 0
recognition_debug = []
current_state = "detecting"  # states: detecting, challenge, recognizing
current_challenge = None
challenge_start_time = 2
challenge_completed = False
face_position_history = []

def get_random_challenge():
    return random.choice(CHALLENGES)

def check_challenge_completion(frame, challenge, face_rect):
    x, y, w, h = face_rect
    face_roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    if challenge == "turn_head":
        # Check if head has moved significantly left or right
        if len(face_position_history) > 10:
            start_x = face_position_history[0][0]
            current_x = x
            if abs(current_x - start_x) > w * 0.3:  # Moved 30% of face width
                return True
                
    elif challenge == "smile":
        # Simple smile detection (could be enhanced with more advanced methods)
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        smiles = smile_cascade.detectMultiScale(gray_roi, scaleFactor=1.8, minNeighbors=20)
        if len(smiles) > 0:
            return True
            
    elif challenge == "blink":
        # Simple blink detection (could be enhanced with eye aspect ratio)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray_roi)
        if len(eyes) < 2:  # Assuming both eyes aren't visible during blink
            return True
            
    elif challenge == "nod":
        # Check if head has moved up or down
        if len(face_position_history) > 10:
            start_y = face_position_history[0][1]
            current_y = y
            if abs(current_y - start_y) > h * 0.2:  # Moved 20% of face height
                return True
                
    return False

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
        current_state = "detecting"
        current_challenge = None
    else:
        # For simplicity, we'll just process the first face found
        (x, y, w, h) = faces[0]
        face_position_history.append((x, y))
        if len(face_position_history) > 20:  # Keep a limited history
            face_position_history.pop(0)
        
        # State machine for the recognition process
        if current_state == "detecting":
            current_challenge = get_random_challenge()
            challenge_start_time = time.time()
            current_state = "challenge"
            recognition_debug.append(f"New challenge: {current_challenge}")
            
        elif current_state == "challenge":
            # Display the challenge instruction
            cv2.putText(frame, f"Please {current_challenge.replace('_', ' ')}", 
                       (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Check if challenge is completed
            if check_challenge_completion(frame, current_challenge, (x, y, w, h)):
                challenge_completed = True
                recognition_debug.append(f"Challenge {current_challenge} completed!")
                current_state = "recognizing"
            elif time.time() - challenge_start_time > CHALLENGE_DURATION:
                recognition_debug.append("Challenge timed out")
                current_state = "detecting"  # Start over
            
        elif current_state == "recognizing":
            # Extract the face ROI and convert to RGB
            face_roi = frame[y:y+h, x:x+w]
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            try:
                # Get 128D encoding for the face
                face_encoding = face_encodings(face_roi_rgb)
                
                if len(face_encoding) == 0:
                    recognition_debug.append("No face encoding generated")
                    current_state = "detecting"
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
                
                # After recognition, go back to detecting
                current_state = "detecting"
            
            except Exception as e:
                recognition_debug.append(f"Error processing face: {str(e)}")
                current_state = "detecting"
                continue
    
    # Display current state and debug info
    cv2.putText(frame, f"State: {current_state}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    for i, msg in enumerate(recognition_debug[-3:]):
        cv2.putText(frame, msg, (10, 60 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()