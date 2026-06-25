import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# --- Incercam sa importam YOLO pentru detectia obiectelor ---
try:
    from ultralytics import YOLO
    
    director_script = os.path.dirname(os.path.abspath(__file__))
    cale_model_custom = os.path.join(director_script, 'model_aparate.pt')
    
    if os.path.exists(cale_model_custom):
        yolo_model = YOLO(cale_model_custom)
        print(f"SUCCES: S-a incarcat modelul tau custom din: {cale_model_custom}")
        model_is_custom = True
    else:
        yolo_model = YOLO('yolov8n.pt') 
        print(f"INFO: Modelul nu a fost gasit la {cale_model_custom}. Se foloseste modelul standard.")
        model_is_custom = False
        
    HAS_YOLO = True
except ImportError:
    yolo_model = None
    HAS_YOLO = False
    model_is_custom = False
    print("AVERTISMENT: Libraria 'ultralytics' nu este instalata.")

# Initializam modulele MediaPipe pentru detectia pozitiei
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculeaza_unghi(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radiani = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    unghi = np.abs(radiani * 180.0 / np.pi)
    if unghi > 180.0: unghi = 360 - unghi
    return unghi

def calculeaza_proiectie_perpendiculara(pivot, p1, p2):
    x0, y0 = pivot
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and y1 == y2: return p1
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    numitor = A**2 + B**2
    if numitor == 0: return p1

    xp = (B * (B * x0 - A * y0) - A * C) / numitor
    yp = (A * (-B * x0 + A * y0) - B * C) / numitor
    return (int(xp), int(yp))

sursa_fortei = None
# Definim variabilele globale pentru hipertrofie la inceput pentru a le accesa in functie
mod_precedent = None
dist_minima_rom = 10000.0
dist_maxima_rom = 0.0
dist_la_tensiune_max = 0.0
tensiune_maxima_inregistrata = 0.0
scor_hipertrofie = "Se calibreaza (Fa o rep)..."

def seteaza_sursa_fortei(event, x, y, flags, param):
    global sursa_fortei
    global dist_minima_rom, dist_maxima_rom, dist_la_tensiune_max, tensiune_maxima_inregistrata, scor_hipertrofie
    
    if event == cv2.EVENT_LBUTTONDOWN:
        sursa_fortei = (x, y)
        # --- NOU: Resetam scorul cand adaugam manual sursa fortei ---
        dist_minima_rom = 10000.0
        dist_maxima_rom = 0.0
        dist_la_tensiune_max = 0.0
        tensiune_maxima_inregistrata = 0.0
        scor_hipertrofie = "Se calibreaza..."
    elif event == cv2.EVENT_RBUTTONDOWN:
        sursa_fortei = None
        # --- NOU: Resetam scorul cand trecem pe gravitatie (stergem manual punctul) ---
        dist_minima_rom = 10000.0
        dist_maxima_rom = 0.0
        dist_la_tensiune_max = 0.0
        tensiune_maxima_inregistrata = 0.0
        scor_hipertrofie = "Se calibreaza..."

def redimensioneaza_cadru(frame, inaltime_tinta=720):
    h, w = frame.shape[:2]
    if h == inaltime_tinta: return frame
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
        cale_fisier = filedialog.askopenfilename(title="Selecteaza media", filetypes=[("Media", "*.mp4;*.avi;*.mov;*.gif")])
        return cale_fisier if cale_fisier else 0
    return 0

sursa = alege_sursa_video()
cap = cv2.VideoCapture(sursa)

cv2.namedWindow('Analiza Biomecanica AI')
cv2.setMouseCallback('Analiza Biomecanica AI', seteaza_sursa_fortei)

is_paused = False

MAPARE_ARTICULATII = {
    'brat_s': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    'brat_d': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    'picior_s': (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    'picior_d': (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
}
NUME_MODURI = {'brat_s': 'Brat Stang', 'brat_d': 'Brat Drept', 'picior_s': 'Picior Stang', 'picior_d': 'Picior Drept'}
PUNCTE_URMARIRE = {k: MAPARE_ARTICULATII[k][2] for k in MAPARE_ARTICULATII}

istoric_miscari = { 'brat_s': [], 'brat_d': [], 'picior_s': [], 'picior_d': [] }
LUNGIME_ISTORIC = 15
auto_mod = True 
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
                    istoric_miscari = {k: [] for k in istoric_miscari}
                    continue
                else: break
                
            if sursa == 0: frame_read = cv2.flip(frame_read, 1)
            frame_read = redimensioneaza_cadru(frame_read, inaltime_tinta=720)
            
        frame = frame_read.copy() 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- LOGICA YOLO ---
        obiect_detectat_yolo = False
        nume_obiect_detectat = ""
        
        if yolo_activat and HAS_YOLO:
            rezultate_yolo = yolo_model(frame, verbose=False, conf=0.70)
            for r in rezultate_yolo:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    nume_obiect = yolo_model.names[cls]
                    
                    if model_is_custom or (nume_obiect in ['bottle', 'cup', 'cell phone', 'laptop', 'mouse']):
                        x1, y1, x2, y2 = box.xyxy[0]
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        sursa_fortei = (cx, cy)
                        obiect_detectat_yolo = True
                        nume_obiect_detectat = nume_obiect
                        
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 3)
                        nume_afisare = f"Scripete ({nume_obiect})" if model_is_custom else f"Obiect ({nume_obiect})"
                        cv2.putText(image, nume_afisare, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        break
        

        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
            
            if not is_paused and auto_mod:
                miscari_curente = {}
                for mod_key, landmark_idx in PUNCTE_URMARIRE.items():
                    punct_curent = [landmarks[landmark_idx.value].x, landmarks[landmark_idx.value].y]
                    istoric_miscari[mod_key].append(punct_curent)
                    
                    if len(istoric_miscari[mod_key]) > LUNGIME_ISTORIC: istoric_miscari[mod_key].pop(0)
                    if len(istoric_miscari[mod_key]) == LUNGIME_ISTORIC:
                        xs = [p[0] for p in istoric_miscari[mod_key]]
                        ys = [p[1] for p in istoric_miscari[mod_key]]
                        miscari_curente[mod_key] = (max(xs) - min(xs)) + (max(ys) - min(ys))

                if miscari_curente:
                    scoruri = {}
                    mod_curent_activ = lista_moduri[index_mod]
                    diag_max = np.sqrt(w**2 + h**2)

                    for mod_key, miscare in miscari_curente.items():
                        scor = miscare  
                        if mod_key == mod_curent_activ:
                            scor *= 5.0 
                        else:
                            if obiect_detectat_yolo and sursa_fortei is not None:
                                px_x = landmarks[PUNCTE_URMARIRE[mod_key].value].x * w
                                px_y = landmarks[PUNCTE_URMARIRE[mod_key].value].y * h
                                dist = np.sqrt((px_x - sursa_fortei[0])**2 + (px_y - sursa_fortei[1])**2)
                                factor_apropiere = max(0.1, (diag_max - dist) / diag_max)
                                scor *= (factor_apropiere ** 3) 
                        scoruri[mod_key] = scor

                    mod_castigator = max(scoruri, key=scoruri.get)
                    if miscari_curente[mod_castigator] > 0.03:
                        index_mod = lista_moduri.index(mod_castigator)
                        
            mod_curent = lista_moduri[index_mod]
            
            # --- RESETARE SCOR DACA SCHIMBAM MEMBRUL ---
            if mod_curent != mod_precedent:
                dist_minima_rom = 10000.0
                dist_maxima_rom = 0.0
                dist_la_tensiune_max = 0.0
                tensiune_maxima_inregistrata = 0.0
                scor_hipertrofie = "Se calibreaza..."
                mod_precedent = mod_curent

            idx_a, idx_b, idx_c = MAPARE_ARTICULATII[mod_curent]
            
            punct_a_px = tuple(np.multiply([landmarks[idx_a.value].x, landmarks[idx_a.value].y], [w, h]).astype(int))
            pivot_px = tuple(np.multiply([landmarks[idx_b.value].x, landmarks[idx_b.value].y], [w, h]).astype(int))
            extremitate_px = tuple(np.multiply([landmarks[idx_c.value].x, landmarks[idx_c.value].y], [w, h]).astype(int))

            if sursa_fortei is not None:
                punct_forta = sursa_fortei
                tip_forta = f"Sursa: Aparat ({nume_obiect_detectat})" if obiect_detectat_yolo else "Sursa: Cablu (Manual)"
                if not obiect_detectat_yolo: cv2.circle(image, sursa_fortei, 15, (0, 165, 255), -1) 
            else:
                punct_forta = (extremitate_px[0], extremitate_px[1] + 1000)
                tip_forta = "Sursa: Gravitatie (Astept scripete...)" if yolo_activat else "Sursa: Gravitatie"
            
            unghi_articulatie = calculeaza_unghi(punct_a_px, pivot_px, extremitate_px)
            unghi_rezistenta = calculeaza_unghi(pivot_px, extremitate_px, punct_forta)
            
            cv2.putText(image, f"Articulatie: {int(unghi_articulatie)} grd", (pivot_px[0] + 15, pivot_px[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f"Rezistenta: {int(unghi_rezistenta)} grd", (extremitate_px[0] + 15, extremitate_px[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            cv2.line(image, extremitate_px, punct_forta, (0, 0, 255), 3) 
            punct_perpendicular = calculeaza_proiectie_perpendiculara(pivot_px, extremitate_px, punct_forta)
            cv2.line(image, pivot_px, punct_perpendicular, (0, 255, 0), 4) 
            
            distanta_d = int(np.sqrt((pivot_px[0] - punct_perpendicular[0])**2 + (pivot_px[1] - punct_perpendicular[1])**2))
            
            d_max = max(1, int(np.sqrt((pivot_px[0] - extremitate_px[0])**2 + (pivot_px[1] - extremitate_px[1])**2)))
            procent_tensiune = min(100, int((distanta_d / d_max) * 100))
            
            # --- CALCUL SCOR HIPERTROFIE (UNIVERSAL) ---
            if not is_paused:
                # 1. Stabilim punctul de ancorare pentru a masura cursa miscarii (ROM)
                if sursa_fortei is not None:
                    ancora_distanta = punct_forta # Pentru scripete, ancora e aparatul
                else:
                    ancora_distanta = (pivot_px[0], h * 2) # Pentru gravitatie, ancora e "pamantul" (punct fix sub utilizator)
                
                # 2. Distanta de la extremitate la ancora
                dist_cablu = np.sqrt((extremitate_px[0] - ancora_distanta[0])**2 + (extremitate_px[1] - ancora_distanta[1])**2)
                
                # 3. Actualizam extremele miscarii
                if dist_cablu < dist_minima_rom:
                    dist_minima_rom = dist_cablu
                if dist_cablu > dist_maxima_rom:
                    dist_maxima_rom = dist_cablu
                
                # 4. Inregistram momentul de tensiune maxima
                if procent_tensiune > tensiune_maxima_inregistrata:
                    tensiune_maxima_inregistrata = procent_tensiune
                    dist_la_tensiune_max = dist_cablu
                
                # 5. Calculam scorul doar daca a existat miscare reala (minim 50 pixeli cursa)
                raza_miscare = dist_maxima_rom - dist_minima_rom
                if raza_miscare > 50.0:
                    pozitie_tensiune = (dist_la_tensiune_max - dist_minima_rom) / raza_miscare
                    nota = max(1.0, min(10.0, 10.0 - (pozitie_tensiune * 6.0)))
                    scor_hipertrofie = f"Nota: {nota:.1f} / 10"

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
            
            # Afisare text Scor Hipertrofie
            cv2.putText(image, "Profil Rezistenta:", (bar_x, bar_y + bar_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, scor_hipertrofie, (bar_x, bar_y + bar_h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if "Nota" in scor_hipertrofie and float(scor_hipertrofie.split(" ")[1]) >= 8 else (0, 165, 255), 2, cv2.LINE_AA)
            
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
                    
                # --- NOU: Resetam scorul hipertrofiei cand schimbam modul YOLO ---
                dist_minima_rom = 10000.0
                dist_maxima_rom = 0.0
                dist_la_tensiune_max = 0.0
                tensiune_maxima_inregistrata = 0.0
                scor_hipertrofie = "Se calibreaza..."
            else:
                print("Libraria ultralytics nu este instalata!")

cap.release()
cv2.destroyAllWindows()