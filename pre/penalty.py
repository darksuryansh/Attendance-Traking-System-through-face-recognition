import face_recognition
import cv2
import pickle
import numpy as np
import random
import time

# Configuration
THRESHOLD = 0.45  # Face recognition threshold
UNKNOWN_NAME = "Unknown"
CHALLENGE_DURATION = 10  # Seconds to complete challenge
PENALTY_DURATION = 15  # Seconds to mark as unknown after failed challenge
CHALLENGE_COOLDOWN = 10  # Seconds between challenges for same person

def load_known_faces(filename):
    """Load known face encodings and names from file"""
    with open(filename, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

def get_challenge():
    """Generate random challenge"""
    challenges = [
        "Blink your eyes",
        "Turn head left",
        "Turn head right",
        "Smile now",
        "Nod your head",
        "Open mouth",
        "Raise eyebrows",
        "Look up",
        "Look down"
    ]
    return random.choice(challenges)

def check_challenge(challenge, prev_frame, current_frame):
    """Verify challenge completion"""
    prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    current_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    
    prev_landmarks = face_recognition.face_landmarks(prev_rgb)
    current_landmarks = face_recognition.face_landmarks(current_rgb)
    
    if not prev_landmarks or not current_landmarks:
        return False
    
    challenge = challenge.lower()
    
    # Eye-related challenges
    if "blink" in challenge:
        prev_eye = get_eye_aspect_ratio(prev_landmarks[0])
        current_eye = get_eye_aspect_ratio(current_landmarks[0])
        return current_eye < prev_eye * 0.7
    
    # Head movement challenges
    elif "left" in challenge:
        prev_nose = np.mean(prev_landmarks[0]['nose_tip'], axis=0)
        current_nose = np.mean(current_landmarks[0]['nose_tip'], axis=0)
        return current_nose[0] > prev_nose[0] + 15
    
    elif "right" in challenge:
        prev_nose = np.mean(prev_landmarks[0]['nose_tip'], axis=0)
        current_nose = np.mean(current_landmarks[0]['nose_tip'], axis=0)
        return current_nose[0] < prev_nose[0] - 15
    
    # Expression challenges
    elif "smile" in challenge:
        return is_smiling(current_landmarks[0]) and not is_smiling(prev_landmarks[0])
    
    elif "nod" in challenge:
        prev_chin = prev_landmarks[0]['chin'][8][1]
        current_chin = current_landmarks[0]['chin'][8][1]
        return current_chin > prev_chin + 8
    
    elif "mouth" in challenge:
        return is_mouth_open(current_landmarks[0]) and not is_mouth_open(prev_landmarks[0])
    
    elif "eyebrows" in challenge:
        return are_eyebrows_raised(current_landmarks[0]) and not are_eyebrows_raised(prev_landmarks[0])
    
    # Gaze challenges
    elif "up" in challenge:
        return is_looking_up(current_landmarks[0]) and not is_looking_up(prev_landmarks[0])
    
    elif "down" in challenge:
        return is_looking_down(current_landmarks[0]) and not is_looking_down(prev_landmarks[0])
    
    return False

def get_eye_aspect_ratio(landmarks):
    """Calculate eye aspect ratio"""
    def eye_ratio(eye):
        v1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        v2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        h = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (v1 + v2) / (2.0 * h)
    left = eye_ratio(landmarks['left_eye'])
    right = eye_ratio(landmarks['right_eye'])
    return (left + right) / 2.0

def is_smiling(landmarks):
    """Check if smiling"""
    top_lip = landmarks['top_lip']
    bottom_lip = landmarks['bottom_lip']
    mouth_width = np.linalg.norm(np.array(top_lip[0]) - np.array(top_lip[6]))
    mouth_height = (max(p[1] for p in bottom_lip) - min(p[1] for p in top_lip))
    return mouth_height > mouth_width * 0.3

def is_mouth_open(landmarks):
    """Check if mouth is open"""
    return get_mouth_height(landmarks) > 20

def get_mouth_height(landmarks):
    """Calculate mouth height"""
    top_lip = landmarks['top_lip']
    bottom_lip = landmarks['bottom_lip']
    return max(p[1] for p in bottom_lip) - min(p[1] for p in top_lip)

def are_eyebrows_raised(landmarks):
    """Check if eyebrows raised"""
    eyes = landmarks['left_eye'] + landmarks['right_eye']
    eyebrows = landmarks['left_eyebrow'] + landmarks['right_eyebrow']
    eye_center = np.mean([p[1] for p in eyes])
    brow_center = np.mean([p[1] for p in eyebrows])
    return brow_center < eye_center - 5

def is_looking_up(landmarks):
    """Check if looking up"""
    nose = landmarks['nose_tip']
    eyes = landmarks['left_eye'] + landmarks['right_eye']
    nose_y = np.mean([p[1] for p in nose])
    eyes_y = np.mean([p[1] for p in eyes])
    return nose_y < eyes_y - 5

def is_looking_down(landmarks):
    """Check if looking down"""
    nose = landmarks['nose_tip']
    eyes = landmarks['left_eye'] + landmarks['right_eye']
    nose_y = np.mean([p[1] for p in nose])
    eyes_y = np.mean([p[1] for p in eyes])
    return nose_y > eyes_y + 5

def recognize_faces(frame, known_encodings, known_names, challenge_status, penalty_list):
    """Recognize faces with challenge system"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    current_time = time.time()
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if this face is in penalty
        in_penalty = False
        for penalty in penalty_list:
            if current_time - penalty['time'] < PENALTY_DURATION:
                penalty_encoding = penalty['encoding']
                distance = face_recognition.face_distance([penalty_encoding], face_encoding)[0]
                if distance < 0.5:  # Similar to the penalized face
                    in_penalty = True
                    break
        
        if in_penalty:
            # Mark as unknown with red rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown (Penalty)", (left, bottom + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            continue
        
        # Normal recognition process
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_idx = np.argmin(face_distances)
        best_distance = face_distances[best_match_idx]
        
        if best_distance <= THRESHOLD:
            name = known_names[best_match_idx]
            color = (0, 255, 0)  # Green
            
            # Check if needs challenge
            needs_challenge = (
                name not in challenge_status or 
                (not challenge_status[name]['active'] and 
                 current_time - challenge_status[name]['last_challenge'] > CHALLENGE_COOLDOWN)
            )
            
            if needs_challenge:
                challenge_status[name] = {
                    'active': True,
                    'text': get_challenge(),
                    'start_time': current_time,
                    'last_challenge': current_time,
                    'prev_frame': frame.copy(),
                    'encoding': face_encoding
                }
        else:
            name = UNKNOWN_NAME
            color = (0, 0, 255)  # Red
        
        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, bottom + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame, challenge_status

def main():
    known_encodings, known_names = load_known_faces("face_encodings.pkl")
    video_capture = cv2.VideoCapture(0)
    
    challenge_status = {}  # {name: {active, text, start_time, last_challenge, prev_frame, encoding}}
    penalty_list = []      # List of faces in penalty
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # Recognize faces and update challenge status
        frame, challenge_status = recognize_faces(
            frame, known_encodings, known_names, challenge_status, penalty_list
        )
        
        # Check active challenges
        for name, status in list(challenge_status.items()):
            if status['active']:
                # Display challenge
                cv2.putText(frame, f"Challenge: {status['text']}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                remaining = CHALLENGE_DURATION - (current_time - status['start_time'])
                cv2.putText(frame, f"Time left: {max(0, round(remaining, 1))}s", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Check challenge completion
                if check_challenge(status['text'], status['prev_frame'], frame):
                    challenge_status[name]['active'] = False
                    print(f"{name} passed challenge!")
                
                # Check timeout
                elif current_time - status['start_time'] > CHALLENGE_DURATION:
                    challenge_status[name]['active'] = False
                    penalty_list.append({
                        'encoding': status['encoding'],
                        'time': current_time
                    })
                    print(f"{name} failed challenge! Penalty active for {PENALTY_DURATION} seconds")
        
        # Cleanup expired penalties
        penalty_list = [p for p in penalty_list if current_time - p['time'] < PENALTY_DURATION]
        
        # Show penalty status
        if penalty_list:
            cv2.putText(frame, f"Security Penalty Active", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()