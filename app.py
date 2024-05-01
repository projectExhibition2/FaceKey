from flask import Flask, render_template, Response, redirect, url_for, request
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
import face_recognition
import cv2
import math
import dlib
import numpy as np
import os
import json
from flask_cors import CORS
from sqlalchemy.exc import IntegrityError

app = Flask(__name__)
CORS(app)

# Configure the database connection URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:45#Purgeon@localhost/faceKeyUsers'

db = SQLAlchemy(app)


# user table
class User(db.Model):
    name = db.Column(db.String(100), nullable = False)
    regNum = db.Column(db.String(20), primary_key=True)
    password = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f'<User {self.name} {self.regNo}>'
    
with app.app_context():
    db.create_all()


# model = pickle.load(open('face_recognition_model.pkl','rb'))
model = pickle.load(open('blink_detection_parameters.pkl','rb'))

# blink detection settings
BLINK_RATIO_THRESHOLD = 5.7

def midpoint(point1, point2):
    return (point1.x + point2.x) / 2, (point1.y + point2.y) / 2

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_blink_ratio(eye_points, facial_landmarks):
    corner_left = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)
    ratio = horizontal_length / vertical_length
    return ratio

#-----Step 3: Face detection with dlib-----
detector = dlib.get_frontal_face_detector()

#-----Step 4: Detecting Eyes using landmarks in dlib-----
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/camera')
def camera():
    return render_template("camera.html")



# @app.route('/face')
# def face_recognition_api():

#     # This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
#     # other example, but it includes some basic performance tweaks to make things run a lot faster:
#     #   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#     #   2. Only detect faces in every other frame of video.

#     # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
#     # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
#     # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

#     # Get a reference to webcam #0 (the default one)
#     video_capture = cv2.VideoCapture(0)

#     # Load a sample picture and learn how to recognize it.
#     obama_image = face_recognition.load_image_file("Abhishek.jpg")
#     obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

#     # Load a second sample picture and learn how to recognize it.
#     biden_image = face_recognition.load_image_file("biden.jpg")
#     biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

#     # Create arrays of known face encodings and their names
#     known_face_encodings = [
#         obama_face_encoding,
#         biden_face_encoding
#     ]
#     known_face_names = [
#         "Abhishek",
#         "Joe Biden"
#     ]

#     # Initialize some variables
#     face_locations = []
#     face_encodings = []
#     face_names = []
#     process_this_frame = True

#     while True:
#         # Grab a single frame of video
#         ret, frame = video_capture.read()

#         # Only process every other frame of video to save time
#         if process_this_frame:
#             # Resize frame of video to 1/4 size for faster face recognition processing
#             small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#             # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#             rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#             # Find all the face locations in the current frame of video
#             face_locations = face_recognition.face_locations(rgb_small_frame)
#             face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#             face_names = []
#             for face_encoding in face_encodings:
#                 # See if the face is a match for the known face(s)
#                 matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                 name = "Unknown"

#                 # # If a match was found in known_face_encodings, just use the first one.
#                 # if True in matches:
#                 #     first_match_index = matches.index(True)
#                 #     name = known_face_names[first_match_index]

#                 # Or instead, use the known face with the smallest distance to the new face
#                 face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#                 best_match_index = np.argmin(face_distances)
#                 if matches[best_match_index]:
#                     name = known_face_names[best_match_index]
#                 # print(name)
#                 face_names.append(name)

#         process_this_frame = not process_this_frame


#         # Display the results
#         for (top, right, bottom, left), name in zip(face_locations, face_names):
#             # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#             top *= 4
#             right *= 4
#             bottom *= 4
#             left *= 4

#             # Draw a box around the face
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#             # Draw a label with a name below the face
#             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#         # Display the resulting image
#         cv2.imshow('Video', frame)

#         # Hit 'q' on the keyboard to quit!
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release handle to the webcam
#     video_capture.release()
#     cv2.destroyAllWindows()

#     # print(face_names)
#     infos = [
#         {
#             "name":"Abhishek",
#             "RegNo": "22BCE10664",
#         },
#         {
#             "name":"Joe Biden",
#             "RegNo":"Kuch to hai"
#         }
#     ]
#     for info in infos:
#         if (name == info["name"]):
#             print(info["name"], info["RegNo"])

#     pythonData = {
#         "name":name
#     }
#     jsonData = json.dumps(pythonData)
#     return jsonData


# def detect_blink():
#     with app.app_context():
#         cap = cv2.VideoCapture(0)
#         cv2.namedWindow('BlinkDetector')
#         blink_count = 0

#         while True:
#             retval, frame = cap.read()

#             if not retval:
#                 print("Can't receive frame (stream end?). Exiting ...")
#                 break 

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             faces, _, _ = detector.run(image=frame, upsample_num_times=0, adjust_threshold=0.0)

#             for face in faces:
#                 landmarks = predictor(frame, face)
#                 left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
#                 right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
#                 blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

#                 if blink_ratio > BLINK_RATIO_THRESHOLD:
#                     cv2.putText(frame, "BLINKING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
#                     blink_count += 1

#             # ret, jpeg = cv2.imencode('.jpg', frame)
#             # frame = jpeg.tobytes()

#             # yield (b'--frame\r\n'
#             #     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#             cv2.imshow('BlinkDetector', frame)
#             key = cv2.waitKey(1)
#             if key & 0XFF==ord('q'):
#                 break


#             if blink_count >= 2:
#                 return render_template('index.html')

#         cap.release()
#         cv2.destroyAllWindows()

# @app.route('/video_feed')
# def video_feed():
#     with app.app_context():
#         cap = cv2.VideoCapture(0)
#         cv2.namedWindow('BlinkDetector')
#         blink_count = 0

#         while True:
#             retval, frame = cap.read()

#             if not retval:
#                 print("Can't receive frame (stream end?). Exiting ...")
#                 break 

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             faces, _, _ = detector.run(image=frame, upsample_num_times=0, adjust_threshold=0.0)

#             for face in faces:
#                 landmarks = predictor(frame, face)
#                 left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
#                 right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
#                 blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

#                 if blink_ratio > BLINK_RATIO_THRESHOLD:
#                     cv2.putText(frame, "BLINKING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    
#                     blink_count += 1

#             # ret, jpeg = cv2.imencode('.jpg', frame)
#             # frame = jpeg.tobytes()

#             # yield (b'--frame\r\n'
#             #     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#             cv2.imshow('BlinkDetector', frame)
#             key = cv2.waitKey(1)
#             if key & 0XFF==ord('q'):
#                 break


#             if blink_count >= 2:
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         return redirect("/")
    
@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate():
    if request.method == 'POST':
        name = request.form['name']
        regNum = request.form['regNum']
        password = request.form['password']

        # Query the database to find the user by registration number
        user = User.query.filter_by(regNum=regNum).first()

        if user:
            # Check if the password matches
            if user.password == password and user.name == name:
                # Authentication successful, redirect to a protected route
                return ("<h1>Authentcation Successfull</h1>")
            else:
                # Password incorrect
                print("Incorrect password or name")
                return redirect('/')
        else:
            # User with the provided registration number does not exist
            print("User not found")
            return redirect('/')

    return render_template('login.html')


@app.route('/matchFace')
def matchFace():
    users = User.query.all()

    # Initialize the infos list
    infos = []

    # Iterate over each user and create the info dictionary
    for user in users:
        info = {
            "name": user.name,
            "RegNo": user.regNum,
            "password": user.password,
            "image_path": user.image_path
        }
        # Append the info dictionary to the infos list
        infos.append(info)

    # Add an "Unknown" entry for cases where the RegNo is empty
    # infos.append({
    #     "name": "unknown",
    #     "RegNo": "",
    #     "password": ""
    # })

    # Print the infos list
    print(infos)

    blink_count = 0
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    known_face_encodings = []
    known_face_names = []

    for info in infos:
        
        # Load a sample picture and learn how to recognize it.
        image = face_recognition.load_image_file(info['image_path'])
        face_encoding = face_recognition.face_encodings(image)[0]

        # Load a second sample picture and learn how to recognize it.
        # biden_image = face_recognition.load_image_file("biden.jpg")
        # biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

        # Create arrays of known face encodings and their names
        known_face_encodings.append(face_encoding)
        known_face_names.append(info["name"])

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the face locations in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                # print(name)
                face_names.append(name)

            faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            for face in faces:
                landmarks = predictor(frame, face)
                left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
                right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
                blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

                if blink_ratio > BLINK_RATIO_THRESHOLD:
                    blink_count += 1

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if blink_count >= 3:
            break
        elif cv2.waitKey(1) & 0xFF == ord('q') :
            name = "Unknown"
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()



    # print(face_names)
    # infos = [
    #     {
    #         "name":"unknown",
    #         "RegNo":"",
    #         "password":""
    #     },
    #     {
    #         "name":"Abhishek",
    #         "RegNo": "22BCE10664",
    #         "password": "Yo@hello"
    #     },
    #     {
    #         "name":"Joe Biden",
    #         "RegNo":"Kuch to hai",
    #         "password":"Kuch to hoga"
    #     }
    # ]
    pythonData = {
        "name":"",
        "RegNo":"",
        "password":""
    }

    for info in infos:
        if (name == info["name"]):
            print(info["name"], info["RegNo"])

            pythonData = {
                "name":name,
                "RegNo": info["RegNo"],
                "password": info["password"]
            }
    
    jsonData = json.dumps(pythonData)
    return jsonData


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        name = request.form['name']
        regNum = request.form['regNum']
        password = request.form['password']
        image_file = request.files['image']

        # Check if a user with the same regNum already exists
        existing_user = User.query.filter_by(regNum=regNum).first()
        if existing_user:
            return 'User with the same registration number already exists'

        
        # Rename the image file to regNum.jpg
        filename = secure_filename(regNum + '.jpg')
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Store the form data in the database
        try:
            user = User(name=name, regNum=regNum, password=password, image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename))
            db.session.add(user)
            db.session.commit()
            return 'Form data and image uploaded successfully'
        except IntegrityError:
            db.session.rollback()
            # If an IntegrityError occurs, it means that the same image_path already exists
            return 'Image with the same name already exists'
        except Exception as e:
            return f'Error: {str(e)}'


    return render_template('upload.html')



# with app.app_context():
    # users = User.query.all()

    # # Initialize the infos list
    # infos = []

    # # Iterate over each user and create the info dictionary
    # for user in users:
    #     info = {
    #         "name": user.name,
    #         "RegNo": user.regNum,
    #         "password": user.password,
    #         "image_path": user.image_path
    #     }
    #     # Append the info dictionary to the infos list
    #     infos.append(info)

    # # Add an "Unknown" entry for cases where the RegNo is empty
    # infos.append({
    #     "name": "unknown",
    #     "RegNo": "",
    #     "password": ""
    # })

    # # Print the infos list
    # print(infos)


if __name__ == "__main__":
    app.run(debug=True)