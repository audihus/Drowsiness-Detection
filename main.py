import cv2
import dlib

detector =  dlib.get_frontal_face_detector() #model untuk mendeteksi wajah
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #model untuk mendeteksi titik wajah

#objek membuat videocapture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #mengubah frame menjadi warna abu
    faces = detector(gray) #deteksi wajah dari frame 

    for face in faces:
        x, y = face.left(), face.top() #mengambil titik koordinat kiri atas wajah yang terdeteksi
        x1, y1 = face.right(), face.bottom() #mengambil titik koordinat kanan bawah wajah yang terdeteksi

        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2) #membuat garis kotak pada frame dari koordinat yang diambil sebelumnya
        #rectangle(frame gambar, koordinat muka kiri atas, koordinat muka kanan bawah, warna, ketebalan)

        landmarks = predictor(gray, face)  #deteksi landmark wajah(mata, hidung, mulut, dsb) pada wajah menggunakan shape_predictor, dengan argumen gray (frame grayscale) dan face (objek muka yang dideteksi)
        
        #membuat gambar 68 titik wajah
        for n in range(68):
            cx = landmarks.part(n).x #koordinat x titik landmarks ke-n 
            cy = landmarks.part(n).y #koordinat y titik landmarks ke-n
            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1) #menggambar titik biru di setiap koordinat
            #circle(frame gambar, koordinat landmark, radius lingkaran, warna, -1 untuk lingkaran penuh)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()