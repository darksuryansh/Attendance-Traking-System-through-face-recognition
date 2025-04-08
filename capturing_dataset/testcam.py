# import cv2
# import os

# # Initialize video capture
# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier('capturing_dataset/haarcascade_frontalface_default.xml')

# # Create directories if they don't exist
# if not os.path.exists('data/images'):
#     os.makedirs('data/images')

# name = input("Enter Your Name: ")

# # Create a directory for the person if it doesn't exist
# person_dir = os.path.join('data/images', name)
# if not os.path.exists(person_dir):
#     os.makedirs(person_dir)

# i = 0
# img_count = 0

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break
        
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
        
#         # Save every 10th frame until we have 100 images
#         if img_count < 100 and i % 10 == 0:
#             # Resize the image to a consistent size
#             resized_img = cv2.resize(crop_img, (200, 200))
            
#             # Save the image
#             img_path = os.path.join(person_dir, f"{name}_{img_count}.jpg")
#             cv2.imwrite(img_path, resized_img)
#             img_count += 1
            
#         i += 1
        
#         # Display the count on the frame
#         cv2.putText(frame, f"Captured: {img_count}/100", (50, 50), 
#                     cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    
#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
    
#     # Exit if 'q' is pressed or we've captured 100 images
#     if k == ord('q') or img_count >= 100:
#         break

# video.release()
# cv2.destroyAllWindows()

# print(f"Successfully captured {img_count} images of {name} in {person_dir}")

import cv2
import os
import pickle
import numpy as np
from face_recognition import face_encodings

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('capturing_dataset/haarcascade_frontalface_default.xml')

# Create directories if they don't exist
if not os.path.exists('data/images'):
    os.makedirs('data/images')

name = input("Enter Your Name: ")

# Create a directory for the person if it doesn't exist
person_dir = os.path.join('data/images', name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

# Initialize lists for encodings
encodings = []
names = []

i = 0
img_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        
        # Save every 10th frame until we have 100 images
        if img_count < 100 and i % 10 == 0:
            # Resize the image to a consistent size
            resized_img = cv2.resize(crop_img, (200, 200))
            
            # Save the image
            img_path = os.path.join(person_dir, f"{name}_{img_count}.jpg")
            cv2.imwrite(img_path, resized_img)
            
            # Generate 128D face encoding
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            try:
                face_encoding = face_encodings(rgb_img)[0]
                encodings.append(face_encoding)
                names.append(name)
            except Exception as e:
                print(f"Couldn't generate encoding for image {img_count}: {str(e)}")
            
            img_count += 1
            
        i += 1
        
        # Display the count on the frame
        cv2.putText(frame, f"Captured: {img_count}/100", (50, 50), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    
    # Exit if 'q' is pressed or we've captured 100 images
    if k == ord('q') or img_count >= 100:
        break

video.release()
cv2.destroyAllWindows()

# Save encodings to pickle file
if len(encodings) > 0:
    # Load existing data if available
    encodings_data = {"encodings": [], "names": []}
    if os.path.exists('data/encodings.pkl'):
        with open('data/encodings.pkl', 'rb') as f:
            encodings_data = pickle.load(f)
    
    # Append new data
    encodings_data["encodings"].extend(encodings)
    encodings_data["names"].extend(names)
    
    # Save updated data
    with open('data/encodings.pkl', 'wb') as f:
        pickle.dump(encodings_data, f)
    
    print(f"Successfully captured {img_count} images and {len(encodings)} encodings of {name}")
else:
    print("No face encodings were generated!")

print(f"Images saved in: {person_dir}")
print(f"Encodings saved in: data/encodings.pkl")
