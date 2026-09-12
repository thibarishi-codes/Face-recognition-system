import face_recognition
import cv2
import numpy as np

# Load your reference image (input image) and encode it
reference_image_path = '/Users/pathin/Desktop/python/pic2.jpeg'  # Replace with your input image file name
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Load the group image (family image) to test and find matches
group_image_path = '/Users/pathin/Desktop/python/pic1.jpeg'  # Replace with your group image file name
group_image = face_recognition.load_image_file(group_image_path)

# Convert the image to RGB (OpenCV uses BGR by default)
group_image_rgb = cv2.cvtColor(group_image, cv2.COLOR_BGR2RGB)

# Detect faces in the group image and get their encodings
face_locations = face_recognition.face_locations(group_image_rgb)
face_encodings = face_recognition.face_encodings(group_image_rgb, face_locations)

# Loop through each face found in the group image
for face_encoding, face_location in zip(face_encodings, face_locations):
    # Compare the face with the reference encoding
    matches = face_recognition.compare_faces([reference_encoding], face_encoding)
    if matches[0]:
        # If a match is found, draw a rectangle around the face
        top, right, bottom, left = face_location
        cv2.rectangle(group_image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            group_image_rgb,
            "Match Found",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    else:
        # Draw rectangles for other unmatched faces
        top, right, bottom, left = face_location
        cv2.rectangle(group_image_rgb, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            group_image_rgb,
            "Not a Match",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

# Show the final output
cv2.imshow("Result", group_image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the output image
cv2.imwrite("output_image.jpg", cv2.cvtColor(group_image_rgb, cv2.COLOR_RGB2BGR))
