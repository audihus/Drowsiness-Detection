from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound
import imutils
import time
import dlib
import cv2
import os

SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
ALARM_SOUND_PATH = os.path.abspath("alertSound.wav")
WEBCAM_INDEX = 0

EYE_AR_THRESH = 0.2 
EYE_AR_CONSEC_FRAMES = 48 
MOUTH_AR_CONSEC_FRAMES = 90 #harus disesuaikan lagi dengan fps video
MOUTH_AR_THRESH = 0.5 

EYE_COUNTER = 0
MOUTH_COUNTER = 0
ALARM_ON = False

BLINK_COUNTER = 0
BLINK_THRESH_PER_MINUTE = 12
BLINK_CONSEC_FRAME = 0

def sound_alarm(path):
    try:
        playsound.playsound(path)
    except Exception as e:
        print(f"[ERROR] Cannot play sound: {e}")

def eye_aspect_ratio(eye):
    A =  dist.euclidean(eye[1], eye[5])
    B =  dist.euclidean(eye[2], eye[4])

    C= dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6]) 
    C = dist.euclidean(mouth[3], mouth[5])

    D =  dist.euclidean(mouth[0], mouth[4]) 
    mar = (A + B + C) / (2.0 * D)
    return mar

def blinkRatePerMinute(duration): #ini kode blink rate
    if duration < 10:
        return 0
    return BLINK_COUNTER/ (duration / 60)

print("[INFO] loading facial landmark predictor...")
detector =  dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=WEBCAM_INDEX).start()
time.sleep(1.0)

start_time = time.time()
lastBlinkCheck = time.time()
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_time = time.time()
    duration = current_time - start_time

    rects = detector(gray,0) 

    for rect in rects: 
        shape = predictor(gray, rect) 
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd] 
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]  
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mar = mouth_aspect_ratio(mouth)
        # blinkRate = blinkRatePerMinute(duration)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull =  cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1,(0,0,255), 1)

        if ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH:
            if ear < EYE_AR_THRESH:
                EYE_COUNTER +=1
            if mar > MOUTH_AR_THRESH:
                MOUTH_COUNTER+=1

            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES or MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True

                    if os.path.exists(ALARM_SOUND_PATH):
                        t =  Thread(target = sound_alarm, args = (ALARM_SOUND_PATH,))
                        t.daemon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
        else :
            EYE_COUNTER = 0
            MOUTH_COUNTER = 0
            ALARM_ON = False

        #ini kode blink rate
        if ear <= 0.13:
            BLINK_CONSEC_FRAME +=1
            if BLINK_CONSEC_FRAME == 1:
                print("Kedipan terdeteksi")
                BLINK_COUNTER += 1 
        else :
            BLINK_CONSEC_FRAME =0

        print(current_time - lastBlinkCheck)
        if current_time - lastBlinkCheck >= 60: #ini kode blink rate
            print("masuk ke if lastBlinkCheeck")
            blink_rate  = BLINK_COUNTER

            if blink_rate < BLINK_THRESH_PER_MINUTE:
                if not ALARM_ON:
                    ALARM_ON = True

                    if os.path.exists(ALARM_SOUND_PATH):
                        t =  Thread(target = sound_alarm, args = (ALARM_SOUND_PATH,))
                        t.daemon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
            else: 
                ALARM_ON = False

            BLINK_COUNTER = 0
            lastBlinkCheck = current_time


        cv2.putText(frame, "EAR: {:.2f}".format(ear),(300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar),(300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        cv2.putText(frame, "Blink : {:.2f}".format(BLINK_COUNTER),(250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        if ALARM_ON:
            print(f"[ALARM] at {time.strftime('%H:%M:%S')} - EAR: {ear:.2f}, MAR: {mar:.2f}")
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # duration = now() - start_time
    #time_per_frame = duration / total_frames

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()