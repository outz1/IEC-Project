import cv2
import time
import mediapipe as mp
import pygame

# ================================
# CONFIGURAÇÕES DO ALERTA
# ================================
ALERT_TIME = 2.5         # segundos com olhos fechados
AFTER_OPEN_TIME = 0.5   # segundos tocando após abrir os olhos
EAR_THRESHOLD = 0.20    # threshold para considerar olho fechado

# ================================
# INICIALIZAÇÃO DO ÁUDIO
# ================================
pygame.mixer.init()
pygame.mixer.music.load("sounds/alert.mp3")

def play_alert_loop():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

def stop_alert():
    pygame.mixer.music.stop()

# ================================
# MEDIAPIPE
# ================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calc_EAR(landmarks, eye):
    v1 = abs(landmarks[eye[1]].y - landmarks[eye[5]].y)
    v2 = abs(landmarks[eye[2]].y - landmarks[eye[4]].y)
    h = abs(landmarks[eye[0]].x - landmarks[eye[3]].x)
    return (v1 + v2) / (2.0 * h)

cap = cv2.VideoCapture(0)

closed_start = None
playing_after_open = False
after_open_start = None
alert_playing = False

print("==========================================")
print("Detector iniciado! Pressione Q para sair.")
print("==========================================")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_ear = calc_EAR(face_landmarks.landmark, LEFT_EYE)
            right_ear = calc_EAR(face_landmarks.landmark, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            def to_pixel(p):
                return int(p.x * w), int(p.y * h)

            left_pts = [to_pixel(face_landmarks.landmark[i]) for i in LEFT_EYE]
            right_pts = [to_pixel(face_landmarks.landmark[i]) for i in RIGHT_EYE]

            lx_min = min(p[0] for p in left_pts)
            lx_max = max(p[0] for p in left_pts)
            ly_min = min(p[1] for p in left_pts)
            ly_max = max(p[1] for p in left_pts)

            rx_min = min(p[0] for p in right_pts)
            rx_max = max(p[0] for p in right_pts)
            ry_min = min(p[1] for p in right_pts)
            ry_max = max(p[1] for p in right_pts)

            eye_closed = ear < EAR_THRESHOLD
            color = (0, 255, 0) if not eye_closed else (0, 0, 255)

            cv2.rectangle(frame, (lx_min, ly_min), (lx_max, ly_max), color, 2)
            cv2.rectangle(frame, (rx_min, ry_min), (rx_max, ry_max), color, 2)

            # ----------------------------------------------------------
            # TEXTO NA TELA
            # ----------------------------------------------------------
            if eye_closed:
                cv2.putText(frame, "Olhos Fechados", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                # Mostrar tempo fechado
                if closed_start is not None:
                    closed_time = time.time() - closed_start
                    cv2.putText(frame, f"{closed_time:.2f}s",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 255), 2)

            else:
                cv2.putText(frame, "Olhos Abertos", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # ----------------------------------------------------------
            # LÓGICA DO ALERTA
            # ----------------------------------------------------------
            if eye_closed:
                if closed_start is None:
                    closed_start = time.time()
                else:
                    if time.time() - closed_start >= ALERT_TIME:
                        play_alert_loop()
                        alert_playing = True
                        playing_after_open = False

            else:
                closed_start = None

                if alert_playing and not playing_after_open:
                    playing_after_open = True
                    after_open_start = time.time()

                if playing_after_open:
                    if time.time() - after_open_start >= AFTER_OPEN_TIME:
                        stop_alert()
                        alert_playing = False
                        playing_after_open = False

    cv2.imshow("Detector de Olhos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_alert()
