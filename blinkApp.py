from flask import Flask, Response, redirect, url_for
import cv2
import dlib
import math

app = Flask(__name__)

BLINK_RATIO_THRESHOLD = 5.7
BLINKS_REQUIRED = 2

def midpoint(point1, point2):
    return (point1.x + point2.x) / 2, (point1.y + point2.y) / 2

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_blink_ratio(eye_points, facial_landmarks):
    corner_left = (facial_landmarks.part(eye_points[0]).x,
                   facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x,
                    facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]),
                          facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                             facial_landmarks.part(eye_points[4]))

    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

blink_count = 0

def detect_blink(frame):
    global blink_count
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, _, _ = detector.run(image=gray, upsample_num_times=0,
                                adjust_threshold=0.0)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if blink_ratio > BLINK_RATIO_THRESHOLD:
            blink_count += 1
            if blink_count >= BLINKS_REQUIRED:
                blink_count = 0
                return True
    return False

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            if detect_blink(frame):
                return redirect(url_for('index'))
            else:
                cv2.putText(frame, "Press 'q' to stop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return 'Home Page'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
