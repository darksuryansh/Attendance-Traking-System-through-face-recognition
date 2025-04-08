import cv2
import os

# Create dataset folder if it doesn't exist
dataset_folder = "dataset"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Input person's name
person_name = input("Enter the person's name: ")
person_folder = os.path.join(dataset_folder, person_name)
if not os.path.exists(person_folder):
    os.makedirs(person_folder)

# Initialize webcam
video_capture = cv2.VideoCapture(0)
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Display the frame
    cv2.imshow('Capturing Images', frame)

    # Save the image when 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        image_path = os.path.join(person_folder, f"{person_name}_{count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")
        count += 1

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()