import cv2
import face_recognition
import os
import numpy as np

known_faces = []
known_names = []

known_faces_dir = "known_faces"
for filename in os.listdir(known_faces_dir):
    img_path = os.path.join(known_faces_dir, filename)
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        
        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_faces, face_encoding))
            name = known_names[match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Live Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

encodings = face_recognition.face_encodings(img)
if encodings:
    encoding = encodings[0]
else:
    print(f"No face found in image: {filename}")

video_capture.release()
cv2.destroyAllWindows()
