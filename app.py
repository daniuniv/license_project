import cv2
import mediapipe as mp
import numpy as np

# Initializam modulele MediaPipe pentru detectia pozitiei
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculeaza_unghi(a, b, c):
    """
    Calculeaza unghiul dintre 3 puncte (ex: Umar, Cot, Incheietura).
    a, b, c sunt tuple/liste de forma [x, y]
    """
    a = np.array(a) # Punct initial (Umar)
    b = np.array(b) # Punct de pivot (Cot - axa de rotatie)
    c = np.array(c) # Punct final (Incheietura - unde actioneaza forta)
    
    # Calculam unghiul in radiani si il transformam in grade
    radiani = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    unghi = np.abs(radiani * 180.0 / np.pi)
    
    if unghi > 180.0:
        unghi = 360 - unghi
        
    return unghi

def calculeaza_proiectie_perpendiculara(pivot, p1, p2):
    """
    Calculeaza proiectia ortogonala a pivotului pe linia fortei (p1 -> p2).
    Asta ne da distanta reala a Bratului Fortei (Moment Arm) in orice directie.
    """
    x0, y0 = pivot
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and y1 == y2:
        return p1

    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    numitor = A**2 + B**2
    if numitor == 0:
        return p1

    xp = (B * (B * x0 - A * y0) - A * C) / numitor
    yp = (A * (-B * x0 + A * y0) - B * C) / numitor

    return (int(xp), int(yp))

# Variabila globala pentru a stoca originea fortei (ex: scripetele)
sursa_fortei = None

def seteaza_sursa_fortei(event, x, y, flags, param):
    global sursa_fortei
    if event == cv2.EVENT_LBUTTONDOWN:
        sursa_fortei = (x, y) # Setam originea la click stanga
    elif event == cv2.EVENT_RBUTTONDOWN:
        sursa_fortei = None   # Resetam la gravitatie cu click dreapta

# Initializam captura video (0 este camera web implicita)
cap = cv2.VideoCapture(0)

# Cream fereastra inainte de loop pentru a atasa evenimentul de mouse
cv2.namedWindow('Analiza Biomecanica AI')
cv2.setMouseCallback('Analiza Biomecanica AI', seteaza_sursa_fortei)

# Setari pentru modelul MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Nu se poate accesa camera.")
            break
            
        # Repozitionam imaginea (oglindire) pentru o utilizare mai naturala
        frame = cv2.flip(frame, 1)
        
        # Convertim imaginea din BGR (standard OpenCV) in RGB (necesar pt MediaPipe)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Procesam imaginea pentru a gasi punctele cheie (landmarks)
        results = pose.process(image)
        
        # Reconversie la BGR pentru a putea desena pe imagine si a o afisa
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extragem coordonatele (daca a detectat un corp)
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Obtinem coordonatele pentru Bratul Stang
            # Inmultim cu latimea si inaltimea imaginii pentru a obtine pixelii exacti
            h, w, _ = image.shape
            
            umar = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            cot = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            
            incheietura = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # 1. Calculam unghiul cotului
            unghi_cot = calculeaza_unghi(umar, cot, incheietura)
            
            # Convertim coordonatele normalizate in pixeli
            cot_px = tuple(np.multiply(cot, [w, h]).astype(int))
            incheietura_px = tuple(np.multiply(incheietura, [w, h]).astype(int))
            umar_px = tuple(np.multiply(umar, [w, h]).astype(int))

            # Afisam unghiul pe ecran, langa cot
            cv2.putText(image, str(int(unghi_cot)) + " grade", 
                        cot_px, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- PARTEA DE BIOMECANICA: CALCUL GENERALIZAT PENTRU ORICE DIRECTIE ---
            
            # Definim punctul catre care trage forta
            if sursa_fortei is not None:
                punct_forta = sursa_fortei
                tip_forta = "Sursa: Cablu/Scripete (Click Dreapta pt Reset)"
                # Desenam sursa fortei pe ecran
                cv2.circle(image, sursa_fortei, 15, (0, 165, 255), -1) 
            else:
                # Default: Gravitatia trage direct in jos
                punct_forta = (incheietura_px[0], incheietura_px[1] + 1000)
                tip_forta = "Sursa: Gravitatie (Click Stg pt a pune un Scripete)"
            
            # 1. Desenam linia de actiune a fortei (cablu sau gravitatie)
            cv2.line(image, incheietura_px, punct_forta, (0, 0, 255), 2)
            
            # 2. Calculam punctul de intersectie perpendicular pentru Moment Arm
            punct_perpendicular = calculeaza_proiectie_perpendiculara(cot_px, incheietura_px, punct_forta)
            
            # 3. Desenam Bratul Fortei (d) perpendicular pe linia de forta
            cv2.line(image, cot_px, punct_perpendicular, (0, 255, 0), 3)
            
            # 4. Calculam distanta (d) in pixeli (ca referinta pentru "Torque")
            distanta_d = int(np.sqrt((cot_px[0] - punct_perpendicular[0])**2 + (cot_px[1] - punct_perpendicular[1])**2))
            
            cv2.putText(image, f"Brat Forta (d): {distanta_d} px", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, tip_forta, 
                        (50, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            
        except Exception as e:
            pass # Daca nu detecteaza pe nimeni, trece peste

        # Desenam toate punctele si legaturile pe corp
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        # Afisam rezultatul
        cv2.imshow('Analiza Biomecanica AI', image)

        # Apasa 'q' pentru a inchide fereastra
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()