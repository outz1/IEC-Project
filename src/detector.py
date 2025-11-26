import cv2
import mediapipe as mp
import time
from playsound import playsound

# Config MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Função que calcula razão de abertura do olho (EAR simplificado)
def eye_aspect_ratio(landmarks, eye_indices):
    # pega pontos importantes
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    # distâncias
    vertical1 = ((p2.x - p6.x)**2 + (p2.y - p6.y)**2)**0.5
    vertical2 = ((p3.x - p5.x)**2 + (p3.y - p5.y)**2)**0.5
    horizontal = ((p1.x - p4.x)**2 + (p1.y - p4.y)**2)**0.5

    ear = (vertical1 + vertical2) / (2 * horizontal)
    return ear

# Índices dos olhos no FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Webcam (AQUI VOCÊ ALTERA SE PRECISAR)
cap = cv2.VideoCapture(0)

eye_closed_threshold = 0.22  # abaixo disso = olho fechado
closed_start = None
ALERT_TIME = 4  # segundos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    eyes_closed = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # EAR dos dois olhos
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)

            ear = (left_ear + right_ear) / 2

            if ear < eye_closed_threshold:
                eyes_closed = True
                cv2.putText(frame, "OLHOS FECHADOS", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Olhos Abertos", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # lógica do timer
    if eyes_closed:
        if closed_start is None:
            closed_start = time.time()
        else:
            elapsed = time.time() - closed_start
            cv2.putText(frame, f"Tempo fechado: {elapsed:.1f}s", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if elapsed >= ALERT_TIME:
                playsound("alert.mp3")
                closed_start = None
    else:
        closed_start = None

    cv2.imshow("Detector de Olhos - MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
