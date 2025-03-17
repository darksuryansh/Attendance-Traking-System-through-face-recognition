import face_recognition
import os
import pickle

# Initialize lists to store encodings and names
known_face_encodings = []
known_face_names = []

# Loop through the dataset
dataset_folder = "preprocessed_dataset"
for person_name in os.listdir(dataset_folder):
    person_folder = os.path.join(dataset_folder, person_name)
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(image_path)

        # Encode the face
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            encoding = face_encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(person_name)

# Save the encodings to a file
with open("face_encodings.pkl", "wb") as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Face encodings saved to face_encodings.pkl")