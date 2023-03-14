import cv2
import dlib

# cargar el detector y el landmark 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# captura video
cap = cv2.VideoCapture(0)

while True:
    # lector de la camara 
    ret, frame = cap.read()

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostro con escala de grieses
    faces = detector(gray)

    # Loop de cada cara
    for face in faces:
        # Obtener el rostro del  landmarks para cada cara 
        landmarks = predictor(gray, face)

        # Dibujar el rectangulo Verde
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Loop encima de la cara con landmarks y dibujar el rostro con los puntos
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Dibujar la linea de los ojo boca y nariz
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        nose = (landmarks.part(30).x, landmarks.part(30).y)
        mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
        mouth_right = (landmarks.part(54).x, landmarks.part(54).y)

        cv2.line(frame, left_eye, (landmarks.part(39).x, landmarks.part(39).y), (0, 255, 0), 2)
        cv2.line(frame, (landmarks.part(39).x, landmarks.part(39).y), (landmarks.part(42).x, landmarks.part(42).y), (0, 255, 0), 2)
        cv2.line(frame, (landmarks.part(42).x, landmarks.part(42).y), right_eye, (0, 255, 0), 2)
        cv2.line(frame, nose, (landmarks.part(33).x, landmarks.part(33).y), (0, 255, 0), 2)
        cv2.line(frame, (landmarks.part(33).x, landmarks.part(33).y), mouth_left, (0, 255, 0), 2)
        cv2.line(frame, mouth_left, (landmarks.part(51).x, landmarks.part(51).y), (0, 255, 0), 2)
        cv2.line(frame, (landmarks.part(51).x, landmarks.part(51).y), mouth_right, (0, 255, 0), 2)
        cv2.line(frame, mouth_right, (landmarks.part(57).x, landmarks.part(57).y), (0, 255, 0), 2)

    # Mostrar el frame 
    cv2.imshow('frame', frame)

    # Para salir apretar "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destruir la pantalla 
cap.release()
cv2.destroyAllWindows()
