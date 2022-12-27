import face_recognition
import cv2
import numpy as np
import time
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
def LoadFaces():
    alex_image = face_recognition.load_image_file('/Users/alexandersuen/PycharmProjects/individualfacialrecognition/Faces/Alex.jpg')
    alex_side_image = face_recognition.load_image_file('/Users/alexandersuen/PycharmProjects/individualfacialrecognition/Faces/alexside.jpg')
    alex_face_encoding = face_recognition.face_encodings(alex_image)[0]
    alex_side_face_encoding = face_recognition.face_encodings(alex_side_image)
    #print(alex_side_face_encoding)
    mom_image = face_recognition.load_image_file('/Users/alexandersuen/PycharmProjects/individualfacialrecognition/Faces/Mom.jpg')
    mom_face_encoding = face_recognition.face_encodings(mom_image)[0]
    dad_image = face_recognition.load_image_file('/Users/alexandersuen/PycharmProjects/individualfacialrecognition/Faces/Dad.jpg')
    dad_face_encoding = face_recognition.face_encodings(dad_image)[0]
    known_face_encodings = [
        alex_face_encoding,
        mom_face_encoding,
        dad_face_encoding
    ]
    #print(known_face_encodings)
    known_face_names = [
        "Alex",
        "Mom",
        "Dad"
    ]
    return known_face_encodings, known_face_names;
known_face_encodings, known_face_names = LoadFaces()
while True:
    start_time = time.time()
    ret, frame = video_capture.read()
    fast_frame = cv2.resize(frame, (0, 0), fx= 0.25, fy=0.25)
    #rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(fast_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(fast_frame, face_locations)
    for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        #matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encodings)
        print(face_distances)
        min_recognize = face_distances.min()
        if min_recognize < 0.46:
            best_match_index = np.argmin(face_distances)
            print(best_match_index)
            name = known_face_names[best_match_index]
        if name == "Alex":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        elif name == "Mom":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        elif name == "Dad":
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (255, 0, 0), cv2.FILLED)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (255, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText (frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
        cv2.putText (frame, str(round(min_recognize, 4)), (right - 6, top - 6), font, 0.7, (255, 255, 255), 1)
    cv2.imshow('Video', frame)
    print("FPS ", 1.0 / (time.time() - start_time))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
