import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# --- NOU: Incercam sa importam YOLO pentru detectia obiectelor ---
try:
    from ultralytics import YOLO
    
    # Cautam modelul tau antrenat, fortand Python sa se uite fix in folderul cu scriptul (app.py)
    director_script = os.path.dirname(os.path.abspath(__file__))
    cale_model_custom = os.path.join(director_script, 'model_aparate.pt')
    
    if os.path.exists(cale_model_custom):
        yolo_model = YOLO(cale_model_custom)
        print(f"SUCCES: S-a incarcat modelul tau custom din: {cale_model_custom}")
        print(f"DEBUG - Clasele modelului tau sunt: {yolo_model.names}") # Vezi in consola cum a numit Roboflow clasa!
        model_is_custom = True
    else:
        # Fallback la modelul standard daca nu ai antrenat inca unul
        yolo_model = YOLO('yolov8n.pt') 
        print(f"INFO: Modelul nu a fost gasit la {cale_model_custom}. Se foloseste modelul standard.")
        model_is_custom = False
        
    HAS_YOLO = True
except ImportError:
    yolo_model = None
    HAS_YOLO = False
    model_is_custom = False
    print("AVERTISMENT: Libraria 'ultralytics' nu este instalata. Ruleaza: pip install ultralytics")

# Initializam modulele MediaPipe pentru detectia pozitiei
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculeaza_unghi(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radiani = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    unghi = np.abs(radiani * 180.0 / np.pi)
    
    if unghi > 180.0:
        unghi = 360 - unghi
        
    return unghi

def calculeaza_proiectie_perpendiculara(pivot, p1, p2):
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

sursa_fortei = None

def seteaza_sursa_fortei(event, x, y, flags, param):
    global sursa_fortei
    if event == cv2.EVENT_LBUTTONDOWN:
        sursa_fortei = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        sursa_fortei = None

def redimensioneaza_cadru(frame, inaltime_tinta=720):
    h, w = frame.shape[:2]
    if h == inaltime_tinta:
        return frame
    raport = inaltime_tinta / float(h)
    latime_noua = int(w * raport)
    return cv2.resize(frame, (latime_noua, inaltime_tinta))

def alege_sursa_video():
    print("="*30)
    print("ANALIZA BIOMECANICA AI")
    print("="*30)
    print("1. Foloseste Camera Web")
    print("2. Incarca un Videoclip sau GIF (.mp4, .avi, .gif)")
    
    alegere = input("Introdu 1 sau 2 si apasa Enter: ")
    
    if alegere == '2':
        root = tk.Tk()
        root.withdraw()
        cale_fisier = filedialog.askopenfilename(
            title="Selecteaza un videoclip sau GIF",
            filetypes=[("Media Files", "*.mp4;*.avi;*.mov;*.gif")]
        )
        if cale_fisier:
            return cale_fisier
        return 0
    return 0

sursa = alege_sursa_video()
cap = cv2.VideoCapture(sursa)

cv2.namedWindow('Analiza Biomecanica AI')
cv2.setMouseCallback('Analiza Biomecanica AI', seteaza_sursa_fortei)

is_paused = False
current_frame = None

MAPARE_ARTICULATII = {
    'brat_s': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    'brat_d': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    'picior_s': (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    'picior_d': (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
}

NUME_MODURI = {
    'brat_s': 'Brat Stang',
    'brat_d': 'Brat Drept',
    'picior_s': 'Picior Stang',
    'picior_d': 'Picior Drept'
}

PUNCTE_URMARIRE = {
    'brat_s': mp_pose.PoseLandmark.LEFT_WRIST,
    'brat_d': mp_pose.PoseLandmark.RIGHT_WRIST,
    'picior_s': mp_pose.PoseLandmark.LEFT_ANKLE,
    'picior_d': mp_pose.PoseLandmark.RIGHT_ANKLE
}

istoric_miscari = { 'brat_s': [], 'brat_d': [], 'picior_s': [], 'picior_d': [] }
LUNGIME_ISTORIC = 15
auto_mod = True 

# Variabila pentru a controla daca YOLO detecteaza automat aparatele
yolo_activat = False

lista_moduri = list(MAPARE_ARTICULATII.keys())
index_mod = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        
        if not is_paused:
            ret, frame_read = cap.read()
            
            if not ret:
                if sursa != 0: 
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    istoric_miscari = { 'brat_s': [], 'brat_d': [], 'picior_s': [], 'picior_d': [] }
                    continue
                else: 
                    break
                
            if sursa == 0:
                frame_read = cv2.flip(frame_read, 1)
            
            frame_read = redimensioneaza_cadru(frame_read, inaltime_tinta=720)
            current_frame = frame_read
            
        # PENTRU YOLO, e cel mai bine sa dam imaginea originala (BGR) asa cum o scoate OpenCV
        frame = current_frame.copy() 
        
        # PENTRU MEDIAPIPE, trebuie neaparat convertita in RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- LOGICA YOLO (DETECTIE APARAT/SCRIPETE) ---
        obiect_detectat_yolo = False
        nume_obiect_detectat = ""
        
        if yolo_activat and HAS_YOLO:
            # Observa: Folosim 'frame' (BGR) aici pentru YOLO
            rezultate_yolo = yolo_model(frame, verbose=False, conf=0.05)
            
            for r in rezultate_yolo:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    nume_obiect = yolo_model.names[cls]
                    
                    # Logica noua: Daca e modelul tau, acceptam orice. Daca e cel standard, cautam anumite obiecte.
                    if model_is_custom or (nume_obiect in ['bottle', 'cup', 'cell phone', 'laptop', 'mouse']):
                        x1, y1, x2, y2 = box.xyxy[0]
                        # Calculam centrul obiectului detectat
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        
                        # Suprascriem sursa fortei catre acest obiect
                        sursa_fortei = (cx, cy)
                        obiect_detectat_yolo = True
                        nume_obiect_detectat = nume_obiect
                        
                        # Desenam bounding box-ul obiectului detectat (scripetelui) pe imaginea finala RGB -> BGR convertita mai jos
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 3)
                        
                        # Afisam frumos in romana
                        nume_afisare = f"Scripete ({nume_obiect})" if model_is_custom else f"Obiect Test ({nume_obiect})"
                        cv2.putText(image, nume_afisare, (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        break # Ne legam de primul obiect valid gasit
        
        # Daca YOLO e activat dar nu vede niciun "aparat", revenim la gravitatie
        if yolo_activat and not obiect_detectat_yolo:
            sursa_fortei = None
        # ----------------------------------------------

        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        
        # Convertim INAPOI in BGR pentru afisare finala corecta pe ecran cu OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
            
            if not is_paused and auto_mod:
                miscari_curente = {}
                for mod_key, landmark_idx in PUNCTE_URMARIRE.items():
                    punct_curent = [landmarks[landmark_idx.value].x, landmarks[landmark_idx.value].y]
                    istoric_miscari[mod_key].append(punct_curent)
                    
                    if len(istoric_miscari[mod_key]) > LUNGIME_ISTORIC:
                        istoric_miscari[mod_key].pop(0)
                    
                    if len(istoric_miscari[mod_key]) == LUNGIME_ISTORIC:
                        xs = [p[0] for p in istoric_miscari[mod_key]]
                        ys = [p[1] for p in istoric_miscari[mod_key]]
                        miscare_totala = (max(xs) - min(xs)) + (max(ys) - min(ys))
                        miscari_curente[mod_key] = miscare_totala

                if miscari_curente:
                    mod_cu_miscare_maxima = max(miscari_curente, key=miscari_curente.get)
                    valoare_maxima = miscari_curente[mod_cu_miscare_maxima]
                    
                    if valoare_maxima > 0.04:
                        index_mod = lista_moduri.index(mod_cu_miscare_maxima)

            mod_curent = lista_moduri[index_mod]
            idx_a, idx_b, idx_c = MAPARE_ARTICULATII[mod_curent]
            
            punct_a = [landmarks[idx_a.value].x, landmarks[idx_a.value].y]
            pivot = [landmarks[idx_b.value].x, landmarks[idx_b.value].y]
            extremitate = [landmarks[idx_c.value].x, landmarks[idx_c.value].y]
            
            pivot_px = tuple(np.multiply(pivot, [w, h]).astype(int))
            extremitate_px = tuple(np.multiply(extremitate, [w, h]).astype(int))
            punct_a_px = tuple(np.multiply(punct_a, [w, h]).astype(int))

            if sursa_fortei is not None:
                punct_forta = sursa_fortei
                
                # Afisam corect sursa fortei in text
                if obiect_detectat_yolo:
                    tip_forta = f"Sursa: Aparat ({nume_obiect_detectat})"
                else:
                    tip_forta = "Sursa: Cablu (Manual)"
                    cv2.circle(image, sursa_fortei, 15, (0, 165, 255), -1) 
            else:
                punct_forta = (extremitate_px[0], extremitate_px[1] + 1000)
                if yolo_activat:
                    tip_forta = "Sursa: Gravitatie (Astept scripete...)"
                else:
                    tip_forta = "Sursa: Gravitatie"
            
            unghi_articulatie = calculeaza_unghi(punct_a_px, pivot_px, extremitate_px)
            unghi_rezistenta = calculeaza_unghi(pivot_px, extremitate_px, punct_forta)
            
            cv2.putText(image, f"Articulatie: {int(unghi_articulatie)} grd", 
                        (pivot_px[0] + 15, pivot_px[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Rezistenta: {int(unghi_rezistenta)} grd", 
                        (extremitate_px[0] + 15, extremitate_px[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

            cv2.line(image, extremitate_px, punct_forta, (0, 0, 255), 3) 
            punct_perpendicular = calculeaza_proiectie_perpendiculara(pivot_px, extremitate_px, punct_forta)
            cv2.line(image, pivot_px, punct_perpendicular, (0, 255, 0), 4) 
            
            distanta_d = int(np.sqrt((pivot_px[0] - punct_perpendicular[0])**2 + (pivot_px[1] - punct_perpendicular[1])**2))
            
            d_max = max(1, int(np.sqrt((pivot_px[0] - extremitate_px[0])**2 + (pivot_px[1] - extremitate_px[1])**2)))
            procent_tensiune = min(100, int((distanta_d / d_max) * 100))
            
            bar_x, bar_y = 50, 240
            bar_w, bar_h = 200, 200 
            
            r = int((procent_tensiune / 100) * 255)
            g = int((1 - procent_tensiune / 100) * 255)
            culoare_tensiune = (0, g, r)
            
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w // 6, bar_y + bar_h), (50, 50, 50), -1)
            inaltime_umplere = int((procent_tensiune / 100) * bar_h)
            cv2.rectangle(image, (bar_x, bar_y + bar_h - inaltime_umplere), (bar_x + bar_w // 6, bar_y + bar_h), culoare_tensiune, -1)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w // 6, bar_y + bar_h), (255, 255, 255), 2)
            
            cv2.putText(image, f"Tensiune: {procent_tensiune}%", 
                        (bar_x, bar_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, culoare_tensiune, 2, cv2.LINE_AA)
            
            cv2.putText(image, f"Brat Forta (d): {distanta_d} px", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, tip_forta, 
                        (50, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            
            if is_paused:
                cv2.putText(image, "PAUZA (Apasa 'P')", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Stare: Ruleaza", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                            
            status_auto = "ON" if auto_mod else "OFF (Manual)"
            cv2.putText(image, f"Mod: {NUME_MODURI[mod_curent]} | Auto: {status_auto}", 
                        (50, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Text status YOLO
            status_yolo = "ON (Cauta obiecte)" if yolo_activat else "OFF"
            cv2.putText(image, f"YOLO Auto-Aparat: {status_yolo}", 
                        (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if yolo_activat else (100, 100, 100), 2, cv2.LINE_AA)
                        
            cv2.putText(image, "Taste: [M] Mod | [A] AutoCorp | [O] YOLO Scripete | [P] Pauza", 
                        (50, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        except Exception as e:
            pass 

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=2))               
        
        cv2.imshow('Analiza Biomecanica AI', image)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') or key == ord(' '): 
            is_paused = not is_paused
        elif key == ord('m'): 
            auto_mod = False 
            index_mod = (index_mod + 1) % len(lista_moduri)
        elif key == ord('a'): 
            auto_mod = not auto_mod
            if auto_mod:
                istoric_miscari = { 'brat_s': [], 'brat_d': [], 'picior_s': [], 'picior_d': [] }
        elif key == ord('o'): # Activam / Dezactivam YOLO
            if HAS_YOLO:
                yolo_activat = not yolo_activat
                if not yolo_activat:
                    sursa_fortei = None # Resetam la gravitatie cand e oprit
            else:
                print("Libraria ultralytics nu este instalata!")

cap.release()
cv2.destroyAllWindows()