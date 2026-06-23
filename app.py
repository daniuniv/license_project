import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Initializam modulele MediaPipe pentru detectia pozitiei
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculeaza_unghi(a, b, c):
    """
    Calculeaza unghiul dintre 3 puncte (ex: Umar, Cot, Incheietura).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radiani = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    unghi = np.abs(radiani * 180.0 / np.pi)
    
    if unghi > 180.0:
        unghi = 360 - unghi
        
    return unghi

def calculeaza_proiectie_perpendiculara(pivot, p1, p2):
    """
    Calculeaza proiectia ortogonala a pivotului pe linia fortei (p1 -> p2).
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

# Variabila globala pentru a stoca originea fortei
sursa_fortei = None

def seteaza_sursa_fortei(event, x, y, flags, param):
    global sursa_fortei
    if event == cv2.EVENT_LBUTTONDOWN:
        sursa_fortei = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        sursa_fortei = None

def redimensioneaza_cadru(frame, inaltime_tinta=720):
    """
    Redimensioneaza imaginea pastrand proportiile, pentru a ne asigura 
    ca interfata (text, bara) are loc mereu pe ecran, indiferent cat de 
    mic sau mare e videoclipul original.
    """
    h, w = frame.shape[:2]
    if h == inaltime_tinta:
        return frame
    raport = inaltime_tinta / float(h)
    latime_noua = int(w * raport)
    return cv2.resize(frame, (latime_noua, inaltime_tinta))

def alege_sursa_video():
    """
    Meniu in terminal pentru a alege intre webcam si fisier video.
    """
    print("="*30)
    print("ANALIZA BIOMECANICA AI")
    print("="*30)
    print("1. Foloseste Camera Web")
    print("2. Incarca un Videoclip sau GIF (.mp4, .avi, .gif)")
    
    alegere = input("Introdu 1 sau 2 si apasa Enter: ")
    
    if alegere == '2':
        # Ascundem fereastra principala a tkinter
        root = tk.Tk()
        root.withdraw()
        
        # Deschidem fereastra de selectie a fisierului
        cale_fisier = filedialog.askopenfilename(
            title="Selecteaza un videoclip sau GIF",
            filetypes=[("Media Files", "*.mp4;*.avi;*.mov;*.gif")]
        )
        
        if cale_fisier:
            print(f"Ai incarcat: {cale_fisier}")
            return cale_fisier
        else:
            print("Nu ai selectat niciun fisier. Se va folosi Camera Web.")
            return 0
    else:
        print("Pornim Camera Web...")
        return 0

# 1. Alegem sursa (0 pentru webcam, cale_fisier pentru video)
sursa = alege_sursa_video()

# 2. Initializam captura
cap = cv2.VideoCapture(sursa)

cv2.namedWindow('Analiza Biomecanica AI')
cv2.setMouseCallback('Analiza Biomecanica AI', seteaza_sursa_fortei)

is_paused = False
current_frame = None

# --- SETARI PENTRU MAI MULTE GRUPE MUSCULARE ---
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

# --- NOU: SETARI PENTRU AUTO-DETECTIE MISCARE ---
# Urmarim doar extremitatile pentru a vedea cine se misca cel mai mult
PUNCTE_URMARIRE = {
    'brat_s': mp_pose.PoseLandmark.LEFT_WRIST,
    'brat_d': mp_pose.PoseLandmark.RIGHT_WRIST,
    'picior_s': mp_pose.PoseLandmark.LEFT_ANKLE,
    'picior_d': mp_pose.PoseLandmark.RIGHT_ANKLE
}

istoric_miscari = { 'brat_s': [], 'brat_d': [], 'picior_s': [], 'picior_d': [] }
LUNGIME_ISTORIC = 15 # Numarul de cadre analizate pentru a stabili miscarea
auto_mod = True # Pilotul automat este pornit implicit

lista_moduri = list(MAPARE_ARTICULATII.keys())
index_mod = 0
# ----------------------------------------------------

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        
        # Daca NU suntem pe pauza, citim un cadru nou
        if not is_paused:
            ret, frame_read = cap.read()
            
            # Daca videoclipul s-a terminat, il reluam de la capat pentru a nu se inchide fereastra
            if not ret:
                if sursa != 0: # Daca este un fisier video/gif
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Resetam la primul cadru
                    istoric_miscari = { 'brat_s': [], 'brat_d': [], 'picior_s': [], 'picior_d': [] } # Golim bufferul de miscare
                    continue
                else: # Daca e camera web si s-a deconectat
                    print("Camera a fost deconectata.")
                    break
                
            # Daca folosim camera web (0), oglindim imaginea.
            if sursa == 0:
                frame_read = cv2.flip(frame_read, 1)
            else:
                pass
            
            # Standardizam dimensiunea cadrului
            frame_read = redimensioneaza_cadru(frame_read, inaltime_tinta=720)
            
            # Salvam cadrul curent
            current_frame = frame_read
            
        # Folosim o copie a cadrului curent pentru a putea recalcula liniile in timp real cat e pus pe pauza
        frame = current_frame.copy()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
            
            # --- LOGICA DE AUTO-DETECTIE A MISCARII ---
            if not is_paused and auto_mod:
                miscari_curente = {}
                for mod_key, landmark_idx in PUNCTE_URMARIRE.items():
                    # Salvam coordonatele x, y curente
                    punct_curent = [landmarks[landmark_idx.value].x, landmarks[landmark_idx.value].y]
                    istoric_miscari[mod_key].append(punct_curent)
                    
                    # Mentinem lista la lungimea maxima dorita (ex: 15 cadre)
                    if len(istoric_miscari[mod_key]) > LUNGIME_ISTORIC:
                        istoric_miscari[mod_key].pop(0)
                    
                    # Calculam distanta maxima parcursa de acest membru in ultimele N cadre
                    if len(istoric_miscari[mod_key]) == LUNGIME_ISTORIC:
                        xs = [p[0] for p in istoric_miscari[mod_key]]
                        ys = [p[1] for p in istoric_miscari[mod_key]]
                        # Suma distantelor pe axele x si y
                        miscare_totala = (max(xs) - min(xs)) + (max(ys) - min(ys))
                        miscari_curente[mod_key] = miscare_totala

                # Daca am strans destule cadre, aflam ce se misca cel mai mult
                if miscari_curente:
                    mod_cu_miscare_maxima = max(miscari_curente, key=miscari_curente.get)
                    valoare_maxima = miscari_curente[mod_cu_miscare_maxima]
                    
                    # Schimbam modul doar daca a existat o miscare semnificativa (ex: > 0.04)
                    # Asta previne schimbarea cursorului cand pur si simplu tremura imaginea
                    if valoare_maxima > 0.04:
                        index_mod = lista_moduri.index(mod_cu_miscare_maxima)
            # ------------------------------------------

            # Preluam modul curent activ (fie ales automat, fie manual)
            mod_curent = lista_moduri[index_mod]
            idx_a, idx_b, idx_c = MAPARE_ARTICULATII[mod_curent]
            
            punct_a = [landmarks[idx_a.value].x, landmarks[idx_a.value].y]
            pivot = [landmarks[idx_b.value].x, landmarks[idx_b.value].y]
            extremitate = [landmarks[idx_c.value].x, landmarks[idx_c.value].y]
            
            pivot_px = tuple(np.multiply(pivot, [w, h]).astype(int))
            extremitate_px = tuple(np.multiply(extremitate, [w, h]).astype(int))
            punct_a_px = tuple(np.multiply(punct_a, [w, h]).astype(int))

            # Determinam originea fortei (pentru desen si calcule)
            if sursa_fortei is not None:
                punct_forta = sursa_fortei
                tip_forta = "Sursa: Cablu/Scripete"
                cv2.circle(image, sursa_fortei, 15, (0, 165, 255), -1) 
            else:
                punct_forta = (extremitate_px[0], extremitate_px[1] + 1000)
                tip_forta = "Sursa: Gravitatie"
            
            # --- CALCUL UNGHIURI ---
            # 1. Unghiul Articulatiei (ex: intre umar, cot si incheietura)
            unghi_articulatie = calculeaza_unghi(punct_a_px, pivot_px, extremitate_px)
            
            # 2. Unghiul de Rezistenta / Forta (intre antebrat si linia de forta)
            # Folosim extremitatea ca varf al unghiului
            unghi_rezistenta = calculeaza_unghi(pivot_px, extremitate_px, punct_forta)
            
            # Afisare Unghi Articulatie (langa pivot/cot) - Culoare Cyan aprins
            cv2.putText(image, f"Articulatie: {int(unghi_articulatie)} grd", 
                        (pivot_px[0] + 15, pivot_px[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Afisare Unghi de Rezistenta (langa extremitate/incheietura) - Culoare Magenta aprins
            cv2.putText(image, f"Rezistenta: {int(unghi_rezistenta)} grd", 
                        (extremitate_px[0] + 15, extremitate_px[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            # -----------------------

            # Desenarea liniilor de actiune (mai groase)
            cv2.line(image, extremitate_px, punct_forta, (0, 0, 255), 3) # Linia fortei - Rosu
            punct_perpendicular = calculeaza_proiectie_perpendiculara(pivot_px, extremitate_px, punct_forta)
            cv2.line(image, pivot_px, punct_perpendicular, (0, 255, 0), 4) # Bratul fortei - Verde gros
            
            # --- CALCUL BRATUL FORTEI SI TENSIUNE ---
            distanta_d = int(np.sqrt((pivot_px[0] - punct_perpendicular[0])**2 + (pivot_px[1] - punct_perpendicular[1])**2))
            
            # Distanta maxima teoretica (lungimea de la pivot la extremitate) pentru a calcula un procent
            d_max = max(1, int(np.sqrt((pivot_px[0] - extremitate_px[0])**2 + (pivot_px[1] - extremitate_px[1])**2)))
            procent_tensiune = min(100, int((distanta_d / d_max) * 100))
            
            # --- DESENARE GRAFIC TENSIUNE ---
            bar_x, bar_y = 50, 220
            bar_w, bar_h = 200, 200 # Inaltimea barei
            
            # Culoare dinamica: Verde la 0%, Rosu la 100% (in BGR)
            r = int((procent_tensiune / 100) * 255)
            g = int((1 - procent_tensiune / 100) * 255)
            culoare_tensiune = (0, g, r)
            
            # Fundal bara
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w // 6, bar_y + bar_h), (50, 50, 50), -1)
            
            # Bara umpluta
            inaltime_umplere = int((procent_tensiune / 100) * bar_h)
            cv2.rectangle(image, (bar_x, bar_y + bar_h - inaltime_umplere), (bar_x + bar_w // 6, bar_y + bar_h), culoare_tensiune, -1)
            
            # Contur bara
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w // 6, bar_y + bar_h), (255, 255, 255), 2)
            
            # Text deasupra barei
            cv2.putText(image, f"Tensiune: {procent_tensiune}%", 
                        (bar_x, bar_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, culoare_tensiune, 2, cv2.LINE_AA)
            # ----------------------------------------
            
            cv2.putText(image, f"Brat Forta (d): {distanta_d} px", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, tip_forta, 
                        (50, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            
            # Adaugam textul pentru starea de pauza
            if is_paused:
                cv2.putText(image, "PAUZA (Apasa 'P' sau Space pt a continua)", 
                            (50, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Stare: Ruleaza", 
                            (50, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                            
            # Afisam grupul muscular analizat curent si statusul Auto
            status_auto = "ON" if auto_mod else "OFF (Manual)"
            cv2.putText(image, f"Mod: {NUME_MODURI[mod_curent]} | Auto: {status_auto}", 
                        (50, 155), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                        
            # Afisare controale / taste valabile
            cv2.putText(image, "Taste: [M] Schimba Mod | [A] Auto On/Off | [P] Pauza", 
                        (50, 185), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        except Exception as e:
            pass 

        # Desenam corpul cu culori NEON foarte vizibile
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4), # Articulatii Rosii
                                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=2)) # Oase Cyan               
        
        cv2.imshow('Analiza Biomecanica AI', image)

        # Citim tastele de la tastatura
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') or key == ord(' '): # Pauza se activeaza cu P sau Space
            is_paused = not is_paused
        elif key == ord('m'): # Schimbam modul (grupa musculara) MANUAL
            auto_mod = False # Dezactivam pilotul automat cand intervine utilizatorul
            index_mod = (index_mod + 1) % len(lista_moduri)
        elif key == ord('a'): # Activam/Dezactivam modul AUTO
            auto_mod = not auto_mod
            # Daca am activat AUTO, resetam istoricul ca sa adune date noi
            if auto_mod:
                istoric_miscari = { 'brat_s': [], 'brat_d': [], 'picior_s': [], 'picior_d': [] }

cap.release()
cv2.destroyAllWindows()