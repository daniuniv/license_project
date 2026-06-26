import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# ==============================================================================
# FUNCTII UTILITARE (MATEMATICA SI GRAFICA)
# ==============================================================================
def calculeaza_unghi(a, b, c):
    """ Calculeaza unghiul format de 3 puncte (ex: umar, cot, incheietura). """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radiani = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    unghi = np.abs(radiani * 180.0 / np.pi)
    return 360 - unghi if unghi > 180.0 else unghi

def calculeaza_proiectie_perpendiculara(pivot, p1, p2):
    """ Calculeaza proiectia pivotului pe vectorul fortei pentru a afla bratul fortei. """
    x0, y0 = pivot
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and y1 == y2: return p1
    A, B = y2 - y1, x1 - x2
    C = x2 * y1 - x1 * y2
    numitor = A**2 + B**2
    if numitor == 0: return p1
    xp = (B * (B * x0 - A * y0) - A * C) / numitor
    yp = (A * (-B * x0 + A * y0) - B * C) / numitor
    return (int(xp), int(yp))

def deseneaza_panel_transparent(img, top_left, bottom_right, culoare=(0, 0, 0), alpha=0.6):
    """ Deseneaza un fundal semi-transparent pentru texte. """
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, culoare, -1)
    cv2.rectangle(overlay, top_left, bottom_right, (100, 100, 100), 1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def afiseaza_text_umbrit(img, text, pozitie, font_scale=0.6, culoare=(255, 255, 255), grosime=2):
    """ Afiseaza text cu umbra pentru lizibilitate maxima. """
    x, y = pozitie
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), grosime + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, culoare, grosime, cv2.LINE_AA)

def redimensioneaza_cadru(frame, inaltime_tinta=720):
    h, w = frame.shape[:2]
    if h == inaltime_tinta: return frame
    raport = inaltime_tinta / float(h)
    return cv2.resize(frame, (int(w * raport), inaltime_tinta))

def alege_sursa_video():
    print("="*30)
    print("ANALIZA BIOMECANICA AI")
    print("="*30)
    print("1. Foloseste Camera Web")
    print("2. Incarca un Videoclip sau GIF")
    alegere = input("Introdu 1 sau 2 si apasa Enter: ")
    if alegere == '2':
        root = tk.Tk()
        root.withdraw()
        cale_fisier = filedialog.askopenfilename(title="Selecteaza media", filetypes=[("Media", "*.mp4;*.avi;*.mov;*.gif")])
        return cale_fisier if cale_fisier else 0
    return 0


# ==============================================================================
# CLASA PRINCIPALA (Arhitectura Orientata pe Obiecte)
# ==============================================================================
class AnalizorBiomecanic:
    def __init__(self):
        # 1. Initializare Modele AI
        self.init_modele_ai()
        
        # 2. Definitii Articulatii MediaPipe
        self.MAPARE_ARTICULATII = {
            'brat_s': (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            'brat_d': (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            'picior_s': (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            'picior_d': (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        }
        self.NUME_MODURI = {'brat_s': 'Brat Stang', 'brat_d': 'Brat Drept', 'picior_s': 'Picior Stang', 'picior_d': 'Picior Drept'}
        self.PUNCTE_URMARIRE = {k: self.MAPARE_ARTICULATII[k][2] for k in self.MAPARE_ARTICULATII}
        self.lista_moduri = list(self.MAPARE_ARTICULATII.keys())
        
        # 3. Stare Aplicatie
        self.is_paused = False
        self.arata_ecran_final = False
        self.auto_mod = True
        self.yolo_activat = False
        self.sursa_fortei = None
        self.istoric_miscari = {k: [] for k in self.MAPARE_ARTICULATII}
        self.index_mod = 0
        self.mod_precedent = None
        
        # 4. Variabile Biomecanica & Hipertrofie
        self.reset_scor()
        
    def reset_scor(self):
        """ Reseteaza contoarele pentru o noua analiza. """
        self.dist_minima_rom = 10000.0
        self.dist_maxima_rom = 0.0
        self.dist_la_tensiune_max = 0.0
        self.tensiune_maxima_inregistrata = 0.0
        self.scor_hipertrofie = "Se calibreaza..."
        self.nota_numerica = 0.0

    def init_modele_ai(self):
        """ Incarca YOLO si MediaPipe. """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        try:
            from ultralytics import YOLO
            director_script = os.path.dirname(os.path.abspath(__file__))
            cale_model_custom = os.path.join(director_script, 'model_aparate.pt')
            
            if os.path.exists(cale_model_custom):
                self.yolo_model = YOLO(cale_model_custom)
                self.model_is_custom = True
            else:
                self.yolo_model = YOLO('yolov8n.pt') 
                self.model_is_custom = False
            self.HAS_YOLO = True
        except ImportError:
            self.yolo_model = None
            self.HAS_YOLO = False
            self.model_is_custom = False

    def callback_mouse(self, event, x, y, flags, param):
        """ Gestioneaza click-urile pe ecran. """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sursa_fortei = (x, y)
            self.reset_scor()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.sursa_fortei = None
            self.reset_scor()

    # ---------------------------------------------------------
    # FUNCTII DE PROCESARE
    # ---------------------------------------------------------
    def detecteaza_sursa_yolo(self, frame_bgr, image_rgb):
        """ Rularea YOLO pentru a gasi scripetele. """
        obiect_detectat = False
        nume_obiect = ""
        box_coords = None

        if self.yolo_activat and self.HAS_YOLO and not self.arata_ecran_final:
            rezultate = self.yolo_model(frame_bgr, verbose=False, conf=0.70)
            for r in rezultate:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    nume = self.yolo_model.names[cls]
                    
                    if self.model_is_custom or (nume in ['bottle', 'cup', 'cell phone']):
                        x1, y1, x2, y2 = box.xyxy[0]
                        self.sursa_fortei = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        obiect_detectat = True
                        nume_obiect = nume
                        box_coords = (int(x1), int(y1), int(x2), int(y2))
                        break
        
        return obiect_detectat, nume_obiect, box_coords

    def identifica_membru_activ(self, landmarks, w, h):
        """ Schimba automat modul pe membrul care se misca cel mai mult. """
        miscari_curente = {}
        for mod_key, landmark_idx in self.PUNCTE_URMARIRE.items():
            punct = [landmarks[landmark_idx.value].x, landmarks[landmark_idx.value].y]
            self.istoric_miscari[mod_key].append(punct)
            
            if len(self.istoric_miscari[mod_key]) > 15:
                self.istoric_miscari[mod_key].pop(0)
            if len(self.istoric_miscari[mod_key]) == 15:
                xs = [p[0] for p in self.istoric_miscari[mod_key]]
                ys = [p[1] for p in self.istoric_miscari[mod_key]]
                miscari_curente[mod_key] = (max(xs) - min(xs)) + (max(ys) - min(ys))

        if miscari_curente:
            scoruri = {}
            mod_curent_activ = self.lista_moduri[self.index_mod]
            diag_max = np.sqrt(w**2 + h**2)

            for mod_key, miscare in miscari_curente.items():
                scor = miscare  
                if mod_key == mod_curent_activ:
                    scor *= 5.0 
                else:
                    if self.sursa_fortei is not None:
                        px_x = landmarks[self.PUNCTE_URMARIRE[mod_key].value].x * w
                        px_y = landmarks[self.PUNCTE_URMARIRE[mod_key].value].y * h
                        dist = np.sqrt((px_x - self.sursa_fortei[0])**2 + (px_y - self.sursa_fortei[1])**2)
                        scor *= (max(0.1, (diag_max - dist) / diag_max) ** 3) 
                scoruri[mod_key] = scor

            mod_castigator = max(scoruri, key=scoruri.get)
            if miscari_curente[mod_castigator] > 0.03:
                self.index_mod = self.lista_moduri.index(mod_castigator)

    def calculeaza_fizica(self, punct_a, pivot, extremitate):
        """ Calculeaza distanta perpendiculara si unghiurile. """
        if self.sursa_fortei is not None:
            punct_forta = self.sursa_fortei
        else:
            punct_forta = (extremitate[0], extremitate[1] + 1000)
            
        unghi_art = calculeaza_unghi(punct_a, pivot, extremitate)
        unghi_rez = calculeaza_unghi(pivot, extremitate, punct_forta)
        
        punct_perp = calculeaza_proiectie_perpendiculara(pivot, extremitate, punct_forta)
        
        dist_d = int(np.sqrt((pivot[0] - punct_perp[0])**2 + (pivot[1] - punct_perp[1])**2))
        d_max = max(1, int(np.sqrt((pivot[0] - extremitate[0])**2 + (pivot[1] - extremitate[1])**2)))
        procent_tens = min(100, int((dist_d / d_max) * 100))
        
        return punct_forta, unghi_art, unghi_rez, punct_perp, dist_d, procent_tens

    def evalueaza_hipertrofia(self, extremitate, punct_forta, procent_tensiune, h):
        """ Calculeaza raza de miscare (ROM) si nota finala a aparatului. """
        ancora = punct_forta if self.sursa_fortei is not None else (extremitate[0], h * 2)
        dist_cablu = np.sqrt((extremitate[0] - ancora[0])**2 + (extremitate[1] - ancora[1])**2)
        
        if dist_cablu < self.dist_minima_rom: self.dist_minima_rom = dist_cablu
        if dist_cablu > self.dist_maxima_rom: self.dist_maxima_rom = dist_cablu
        
        if procent_tensiune > self.tensiune_maxima_inregistrata:
            self.tensiune_maxima_inregistrata = procent_tensiune
            self.dist_la_tensiune_max = dist_cablu
        
        raza_miscare = self.dist_maxima_rom - self.dist_minima_rom
        if raza_miscare > 50.0:
            pozitie_tensiune = (self.dist_la_tensiune_max - self.dist_minima_rom) / raza_miscare
            self.nota_numerica = max(1.0, min(10.0, 10.0 - (pozitie_tensiune * 6.0)))
            self.scor_hipertrofie = f"Scor: {self.nota_numerica:.1f} / 10"

    # ---------------------------------------------------------
    # FUNCTII DESENARE UI
    # ---------------------------------------------------------
    def deseneaza_grafica_biomecanica(self, image, extremitate, pivot, punct_forta, punct_perp, unghi_art, unghi_rez):
        """ Traseaza vectorii pe schelet. """
        cv2.putText(image, f"Articulatie: {int(unghi_art)} grd", (pivot[0] + 15, pivot[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Rezistenta: {int(unghi_rez)} grd", (extremitate[0] + 15, extremitate[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(image, extremitate, punct_forta, (0, 0, 255), 3) 
        cv2.line(image, pivot, punct_perp, (0, 255, 0), 4)

    def deseneaza_hud_principal(self, image, tip_forta, procent_tensiune, dist_d, h, w):
        """ Deseneaza panourile de informatii din stanga si dreapta. """
        # --- PANOU STANGA SUS ---
        # Am extins inaltimea de la 150 la 180 pentru a face loc randului nou
        deseneaza_panel_transparent(image, (15, 15), (380, 180), (20, 20, 20), 0.7)
        
        status_sistem = "PAUZA" if self.is_paused else "ACTIV"
        culoare_sys = (0, 0, 255) if self.is_paused else (0, 255, 0)
        afiseaza_text_umbrit(image, f"SISTEM: {status_sistem}", (30, 45), 0.7, culoare_sys, 2)
        
        mod_afisaj = self.NUME_MODURI[self.lista_moduri[self.index_mod]]
        afiseaza_text_umbrit(image, f"Membru: {mod_afisaj} [{'Auto' if self.auto_mod else 'Manual'}]", (30, 75), 0.6, (255, 255, 0), 1)
        
        status_yolo = "ON (Cauta)" if self.yolo_activat else "OFF"
        culoare_yolo = (0, 255, 255) if self.yolo_activat else (150, 150, 150)
        afiseaza_text_umbrit(image, f"YOLO AI: {status_yolo}", (30, 105), 0.6, culoare_yolo, 1)
        afiseaza_text_umbrit(image, f"Sursa: {tip_forta}", (30, 135), 0.5, (0, 165, 255), 1)
        
        # Am mutat Bratul Fortei ordonat aici jos in cadrul panoului
        afiseaza_text_umbrit(image, f"Brat Forta (d): {dist_d} px", (30, 165), 0.6, (0, 255, 0), 1)

        # --- PANOU BARA TENSIUNE (Stanga Jos) ---
        deseneaza_panel_transparent(image, (15, 420), (280, 700), (20, 20, 20), 0.7)
        bar_y, bar_w, bar_h = 460, 40, 150
        bar_x = 50
        r = int((procent_tensiune / 100) * 255)
        g = int((1 - procent_tensiune / 100) * 255)
        
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(image, (bar_x, bar_y + bar_h - int((procent_tensiune/100)*bar_h)), (bar_x + bar_w, bar_y + bar_h), (0, g, r), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        afiseaza_text_umbrit(image, "TENSIUNE", (bar_x - 15, bar_y - 10), 0.5, (200, 200, 200), 1)
        afiseaza_text_umbrit(image, f"{procent_tensiune}%", (bar_x - 5, bar_y + bar_h + 20), 0.5, (0, g, r), 2)
        
        # Scor Text
        culoare_scor = (0, 255, 0) if self.nota_numerica >= 8 else ((0, 255, 255) if self.nota_numerica >= 5 else (0, 0, 255))
        if "calibreaza" in self.scor_hipertrofie: culoare_scor = (200, 200, 200)
        afiseaza_text_umbrit(image, "Evaluare Profil:", (30, 650), 0.6, (255, 255, 255), 1)
        afiseaza_text_umbrit(image, self.scor_hipertrofie, (30, 680), 0.8, culoare_scor, 2)

        # --- PANOU AJUTOR (Dreapta Jos) ---
        deseneaza_panel_transparent(image, (w - 380, h - 130), (w - 15, h - 15), (20, 20, 20), 0.5)
        afiseaza_text_umbrit(image, "[M] Mod  | [A] AutoMembru", (w - 365, h - 105), 0.5, (200, 200, 200), 1)
        afiseaza_text_umbrit(image, "[O] YOLO | [P] Pauza", (w - 365, h - 80), 0.5, (200, 200, 200), 1)
        afiseaza_text_umbrit(image, "[Click] Punct  | [Click Dr] Sterge", (w - 365, h - 55), 0.5, (200, 200, 200), 1)
        afiseaza_text_umbrit(image, "[E] ECRAN EVALUARE FINALA", (w - 365, h - 30), 0.6, (0, 255, 255), 2)

    def deseneaza_ecran_evaluare(self, image):
        """ Afiseaza caseta centrala cand apesi tasta E. """
        h, w = image.shape[:2]
        deseneaza_panel_transparent(image, (0, 0), (w, h), (10, 10, 15), 0.85)
        
        caseta_w, caseta_h = 600, 400
        sx, sy = (w - caseta_w) // 2, (h - caseta_h) // 2
        cv2.rectangle(image, (sx, sy), (sx + caseta_w, sy + caseta_h), (30, 30, 35), -1)
        cv2.rectangle(image, (sx, sy), (sx + caseta_w, sy + caseta_h), (0, 165, 255), 2)
        
        afiseaza_text_umbrit(image, "RAPORT EVALUARE APARAT", (sx + 90, sy + 60), 1.0, (255, 255, 255), 3)
        cv2.line(image, (sx + 50, sy + 80), (sx + caseta_w - 50, sy + 80), (100, 100, 100), 2)
        
        clr_scor = (0, 255, 0) if self.nota_numerica >= 8 else ((0, 255, 255) if self.nota_numerica >= 5 else (0, 0, 255))
        verdict = "Verdict: EXCELENT" if self.nota_numerica >= 8 else ("Verdict: ACCEPTABIL" if self.nota_numerica >= 5 else "Verdict: SUB-OPTIM")
        
        if "calibreaza" in self.scor_hipertrofie or (self.dist_maxima_rom - self.dist_minima_rom) <= 50.0: 
            clr_scor = (150, 150, 150)
            verdict = "Verdict: DATE INSUFICIENTE (Fa o repetare intreaga)"
            
        afiseaza_text_umbrit(image, f"SCOR FINAL: {self.nota_numerica:.1f} / 10", (sx + 120, sy + 180), 1.2, clr_scor, 3)
        afiseaza_text_umbrit(image, verdict, (sx + 160, sy + 230), 0.7, clr_scor, 2)
        afiseaza_text_umbrit(image, "Apasa 'E' pentru a reveni la analiza", (sx + 130, sy + 360), 0.6, (150, 150, 150), 1)

    # ---------------------------------------------------------
    # BUCLA PRINCIPALA A APLICATIEI
    # ---------------------------------------------------------
    def ruleaza(self, sursa_video):
        cap = cv2.VideoCapture(sursa_video)
        cv2.namedWindow('Analiza Biomecanica AI')
        cv2.setMouseCallback('Analiza Biomecanica AI', self.callback_mouse)
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                if not self.is_paused and not self.arata_ecran_final:
                    ret, frame_read = cap.read()
                    if not ret:
                        if sursa_video != 0: 
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.istoric_miscari = {k: [] for k in self.MAPARE_ARTICULATII}
                            continue
                        else: break
                        
                    if sursa_video == 0: frame_read = cv2.flip(frame_read, 1)
                    frame_read = redimensioneaza_cadru(frame_read, inaltime_tinta=720)
                    
                frame = frame_read.copy() 
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 1. AI: Detectie Obiecte (YOLO)
                yolo_gasit, nume_obj, box_coords = self.detecteaza_sursa_yolo(frame, image_rgb)
                
                if yolo_gasit:
                    x1, y1, x2, y2 = box_coords
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    titlu = f"Scripete ({nume_obj})" if self.model_is_custom else f"Obiect ({nume_obj})"
                    afiseaza_text_umbrit(image_rgb, titlu, (x1, y1-10), 0.5, (0, 165, 255), 1)
                elif self.yolo_activat and not yolo_gasit:
                    self.sursa_fortei = None

                # 2. AI: Analiza Postura (MediaPipe)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                tip_forta = "Se incarca..."
                procent_tens, dist_d = 0, 0
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = image_bgr.shape
                    
                    # Logica Schimbare Mod
                    if not self.is_paused and self.auto_mod and not self.arata_ecran_final:
                        self.identifica_membru_activ(landmarks, w, h)
                        
                    mod_curent = self.lista_moduri[self.index_mod]
                    if mod_curent != self.mod_precedent:
                        self.reset_scor()
                        self.mod_precedent = mod_curent

                    # Extractie Coordonate
                    idx_a, idx_b, idx_c = self.MAPARE_ARTICULATII[mod_curent]
                    pt_a = tuple(np.multiply([landmarks[idx_a.value].x, landmarks[idx_a.value].y], [w, h]).astype(int))
                    pivot = tuple(np.multiply([landmarks[idx_b.value].x, landmarks[idx_b.value].y], [w, h]).astype(int))
                    extrem = tuple(np.multiply([landmarks[idx_c.value].x, landmarks[idx_c.value].y], [w, h]).astype(int))

                    # 3. Calcul Biomecanic
                    punct_forta, unghi_art, unghi_rez, punct_perp, dist_d, procent_tens = self.calculeaza_fizica(pt_a, pivot, extrem)
                    
                    if self.sursa_fortei is not None:
                        tip_forta = f"Aparat ({nume_obj})" if yolo_gasit else "Cablu (Manual)"
                        if not yolo_gasit: cv2.circle(image_bgr, self.sursa_fortei, 10, (0, 165, 255), 2)
                    else:
                        tip_forta = "Gravitatie (Astept scripete...)" if self.yolo_activat else "Gravitatie"

                    # 4. Hipertrofie
                    if not self.is_paused and not self.arata_ecran_final:
                        self.evalueaza_hipertrofia(extrem, punct_forta, procent_tens, h)

                    # 5. Desenare
                    if not self.arata_ecran_final:
                        self.deseneaza_grafica_biomecanica(image_bgr, extrem, pivot, punct_forta, punct_perp, unghi_art, unghi_rez)
                        self.deseneaza_hud_principal(image_bgr, tip_forta, procent_tens, dist_d, h, w)
                
                except Exception:
                    pass 

                # Suprapunere schelet MediaPipe
                if not self.arata_ecran_final:
                    self.mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4), 
                                            self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=2))               

                # 6. Ecran Final
                if self.arata_ecran_final:
                    self.deseneaza_ecran_evaluare(image_bgr)

                cv2.imshow('Analiza Biomecanica AI', image_bgr)

                # ==========================================
                # EVENT-URI TASTATURA
                # ==========================================
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'): break
                elif key == ord('e'): 
                    self.arata_ecran_final = not self.arata_ecran_final
                    self.is_paused = self.arata_ecran_final
                elif key == ord('p') or key == ord(' '): 
                    if not self.arata_ecran_final: self.is_paused = not self.is_paused
                elif key == ord('m') and not self.arata_ecran_final: 
                    self.auto_mod = False 
                    self.index_mod = (self.index_mod + 1) % len(self.lista_moduri)
                elif key == ord('a') and not self.arata_ecran_final: 
                    self.auto_mod = not self.auto_mod
                    if self.auto_mod: self.istoric_miscari = {k: [] for k in self.MAPARE_ARTICULATII}
                elif key == ord('o') and not self.arata_ecran_final: 
                    if self.HAS_YOLO:
                        self.yolo_activat = not self.yolo_activat
                        if not self.yolo_activat: self.sursa_fortei = None 
                        self.reset_scor()
                    else:
                        print("Libraria ultralytics nu este instalata!")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sursa_selectata = alege_sursa_video()
    aplicatie = AnalizorBiomecanic()
    aplicatie.ruleaza(sursa_selectata)