import cv2
import os

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to preprocess an image
def preprocess_image(image_path, output_path, target_size=(160, 160)):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Convert to grayscale (optional but recommended)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return

    # Assume the first face is the target
    (x, y, w, h) = faces[0]

    # Crop the face
    face = gray[y:y+h, x:x+w]

    # Resize the face to the target size
    resized = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)

    # Save the preprocessed image
    cv2.imwrite(output_path, resized)
    print(f"Processed and saved {output_path}")

# Function to preprocess the entire dataset
def preprocess_dataset(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for person_name in os.listdir(input_folder):
        person_folder = os.path.join(input_folder, person_name)
        output_person_folder = os.path.join(output_folder, person_name)
        if not os.path.exists(output_person_folder):
            os.makedirs(output_person_folder)

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            output_path = os.path.join(output_person_folder, image_name)
            preprocess_image(image_path, output_path)

# Preprocess the dataset
input_folder = "dataset"  # Folder containing raw images
output_folder = "preprocessed_dataset"  # Folder to save preprocessed images
preprocess_dataset(input_folder, output_folder)