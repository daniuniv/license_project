import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import time
import customtkinter as ctk

# ==============================================================================
# FUNCTII UTILITARE (MATEMATICA SI GRAFICA)
# ==============================================================================
def calculeaza_unghi(a, b, c):
    """ Calculeaza unghiul format de 3 puncte tridimensionale sau bidimensionale. """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radiani = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    unghi = np.abs(radiani * 180.0 / np.pi)
    return 360 - unghi if unghi > 180.0 else unghi

def calculeaza_proiectie_perpendiculara(pivot, p1, p2):
    """ Calculeaza proiectia pivotului pe vectorul fortei pentru a afla bratul fortei (d). """
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
    """ Deseneaza un fundal semi-transparent pentru lizibilitatea textelor din HUD. """
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, culoare, -1)
    cv2.rectangle(overlay, top_left, bottom_right, (100, 100, 100), 1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def afiseaza_text_umbrit(img, text, pozitie, font_scale=0.6, culoare=(255, 255, 255), grosime=2):
    """ Afiseaza text cu umbra neagra de contrast pentru a fi lizibil pe orice fundal. """
    x, y = pozitie
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), grosime + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, culoare, grosime, cv2.LINE_AA)

def redimensioneaza_cadru(frame, inaltime_tinta=720):
    """ Redimensioneaza cadrul pastrand proportiile pentru a nu deforma analiza. """
    h, w = frame.shape[:2]
    if h == inaltime_tinta: return frame
    raport = inaltime_tinta / float(h)
    return cv2.resize(frame, (int(w * raport), inaltime_tinta))

# ==============================================================================
# CLASA PRINCIPALA DE ANALIZA BIOMECANICA
# ==============================================================================
class AnalizorBiomecanic:
    def __init__(self):
        self.init_modele_ai()
        
        # Harta articulatiilor cheie: (Articulatie de referinta, Pivot / Centru de rotatie, Extremitate mobila)
        self.MAPARE_ARTICULATII = {
            'brat_s': (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            'brat_d': (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            'picior_s': (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            'picior_d': (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            'umar_s': (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            'umar_d': (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW)
        }
        
        self.NUME_MODURI = {
            'brat_s': 'Brat Stang', 
            'brat_d': 'Brat Drept', 
            'picior_s': 'Picior Stang', 
            'picior_d': 'Picior Drept',
            'umar_s': 'Umar Stang',
            'umar_d': 'Umar Drept'
        }
        
        self.PUNCTE_URMARIRE = {
            'brat_s': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'brat_d': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'picior_s': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'picior_d': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'umar_s': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'umar_d': self.mp_pose.PoseLandmark.RIGHT_ELBOW
        }
        
        self.lista_moduri = list(self.MAPARE_ARTICULATII.keys())
        
        self.is_paused = False
        self.arata_ecran_final = False
        self.auto_mod = True
        self.yolo_activat = True
        self.sursa_fortei = None
        self.istoric_miscari = {k: [] for k in self.MAPARE_ARTICULATII}
        self.index_mod = 0
        self.mod_precedent = None
        
        self.ultima_pozitie_yolo = None
        self.timp_ultima_detectie = 0
        self.timeout_detectie = 130.0  
        
        self.timp_ultima_miscare_activa = time.time()
        self.perioada_gratie_switch = 2.0 
        
        # Flag pentru a evita resetarea cand scripetele dispare si reapare
        self.calibrare_initiala_yolo_facuta = False 
        
        self.reset_scor()
        
    def reset_scor(self):
        """ Reseteaza toate datele de telemetrie, profilul si executia. """
        self.dist_minima_rom = 10000.0
        self.dist_maxima_rom = 0.0
        self.dist_la_tensiune_max = 0.0
        self.tensiune_maxima_inregistrata = 0.0
        self.scor_hipertrofie = "Se calibreaza..."
        self.nota_numerica = 0.0
        self.tip_forta = "Se incarca..."
        
        # State machine pentru clasificarea exercitiilor pe baza posturii initiale
        self.unghi_start = None
        self.unghi_aux_start = None
        self.exercitiu_detectat = "Asteptare miscare..."
        self.prag_detectie = 15.0 # Diferenta minima de unghi pentru a confirma directia
        
        # --- NOU: Parametrii Form Check (Scor Executie) ---
        self.scor_executie = 10.0
        self.repetari_analizate = 0
        self.mesaj_form_check = "Astept prima repetare..."
        self.stadiu_repetare = "asteptare" # asteptare, concentric, excentric
        self.unghi_minim_curent = 999.0
        self.unghi_maxim_curent = 0.0
        self.unghi_aux_minim = 999.0
        self.unghi_aux_maxim = 0.0
        self.timp_start_faza = time.time()
        self.istoric_form_unghi = []
        self.unghiuri_start_buffer = [] # NOU: Colecție pentru media unghiurilor

    def init_modele_ai(self):
        """ Initializeaza modelele AI pentru estimarea corpului (MediaPipe) si detectarea scripetilor (YOLO). """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        director_script = os.path.dirname(os.path.abspath(__file__))
        
        try:
            from ultralytics import YOLO
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
        """ Permite plasarea manuala a sursei fortei (ex: scripete, disc) cu Click-Stanga si stergerea cu Click-Dreapta. """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sursa_fortei = (x, y)
            self.ultima_pozitie_yolo = None 
            self.reset_scor()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.sursa_fortei = None
            self.ultima_pozitie_yolo = None 
            self.reset_scor()

    # ==============================================================================
    # LOGICA DE DETECTIE SI CLASIFICARE ML / GEOMETRICA
    # ==============================================================================
    def detecteaza_sursa_yolo(self, frame_bgr, image_rgb):
        """ Utilizeaza YOLO pentru a detecta obiectele de fitness care pot reprezenta ancora fortei. """
        obiect_detectat_acum = False
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
                        noua_pozitie = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        if self.ultima_pozitie_yolo:
                            alfa = 0.2  
                            x_smooth = int(self.ultima_pozitie_yolo[0] * (1 - alfa) + noua_pozitie[0] * alfa)
                            y_smooth = int(self.ultima_pozitie_yolo[1] * (1 - alfa) + noua_pozitie[1] * alfa)
                            self.sursa_fortei = (x_smooth, y_smooth)
                        else:
                            self.sursa_fortei = noua_pozitie
                            
                        self.ultima_pozitie_yolo = self.sursa_fortei
                        self.timp_ultima_detectie = time.time()
                        
                        obiect_detectat_acum = True
                        nume_obiect = nume
                        box_coords = (int(x1), int(y1), int(x2), int(y2))
                        break
        
        if not obiect_detectat_acum and self.yolo_activat:
            if self.timp_ultima_detectie != 0 and (time.time() - self.timp_ultima_detectie < self.timeout_detectie):
                pass
            else:
                self.sursa_fortei = None
                self.ultima_pozitie_yolo = None
        
        return obiect_detectat_acum, nume_obiect, box_coords

    def identifica_membru_activ(self, landmarks, w, h):
        """ Identifica automat care este membrul in miscare bazandu-se pe VARIATIA UNGHIULARA a articulatiilor (grade). """
        miscari_curente = {}
        visibilitate_buna = {}
        
        fereastra_cadre = 25 
        
        for mod_key in self.lista_moduri:
            idx_a, idx_b, idx_c = self.MAPARE_ARTICULATII[mod_key]
            vis_a = landmarks[idx_a.value].visibility
            vis_b = landmarks[idx_b.value].visibility
            vis_c = landmarks[idx_c.value].visibility
            
            visibilitate_buna[mod_key] = (vis_a > 0.5 and vis_b > 0.5 and vis_c > 0.5)

            pt_a = [landmarks[idx_a.value].x, landmarks[idx_a.value].y]
            pt_b = [landmarks[idx_b.value].x, landmarks[idx_b.value].y]
            pt_c = [landmarks[idx_c.value].x, landmarks[idx_c.value].y]
            
            unghi_curent = calculeaza_unghi(pt_a, pt_b, pt_c)
            self.istoric_miscari[mod_key].append(unghi_curent)
            
            if len(self.istoric_miscari[mod_key]) > fereastra_cadre: 
                self.istoric_miscari[mod_key].pop(0)
            
            if len(self.istoric_miscari[mod_key]) == fereastra_cadre:
                if not visibilitate_buna[mod_key]:
                    miscari_curente[mod_key] = 0.0
                else:
                    istoric_sortat = sorted(self.istoric_miscari[mod_key])
                    # Eliminam cele 2 valori de sus si de jos pt a ignora glitch-urile de tracking
                    fara_outliers = istoric_sortat[2:-2] if len(istoric_sortat) > 5 else istoric_sortat
                    miscari_curente[mod_key] = max(fara_outliers) - min(fara_outliers)

        if not miscari_curente:
            return

        # 1. Analizam membrul CURENT ACTIV
        mod_curent_activ = self.lista_moduri[self.index_mod]
        miscare_curenta = miscari_curente.get(mod_curent_activ, 0.0)
        
        # Daca membrul activ se misca peste nivelul de zgomot, actualizam timer-ul (Lock-In)
        if miscare_curenta > 4.0 and visibilitate_buna.get(mod_curent_activ, False):
            self.timp_ultima_miscare_activa = time.time()
            return # <--- ACESTA ESTE RETURN-UL SALVATOR! Fara el forta un reset la fiecare cadru.
            
        timp_expirat = (time.time() - self.timp_ultima_miscare_activa) > self.perioada_gratie_switch

        # 2. Analizam CELELALTE membre (candidatii) izoland complet membrul activ curent
        alte_membre_in_miscare = {k: v for k, v in miscari_curente.items() if k != mod_curent_activ and visibilitate_buna.get(k, False)}
        
        if not alte_membre_in_miscare:
            return
            
        cel_mai_activ_alt_mod = max(alte_membre_in_miscare, key=alte_membre_in_miscare.get)
        variatie_alt_mod = alte_membre_in_miscare[cel_mai_activ_alt_mod]

        # 3. Luam decizia de a SCHIMBA FOCUSUL
        vrem_sa_schimbam = False
        
        # PREVENTIE FLICKER / RESET FALS:
        # Daca am detectat deja un exercitiu (ex: Flexii Biceps), IGNORAM orice spike-uri de miscare false 
        # (ex: frame drop-urile si glitch-urile cand YOLO re-detecteaza scripetele pe ecran)
        if self.exercitiu_detectat != "Asteptare miscare...":
            # Schimbam focusul STRICT DOAR daca utilizatorul s-a oprit complet (a expirat gratia) si alt membru se misca mult
            if timp_expirat and variatie_alt_mod > 10.0:
                vrem_sa_schimbam = True
        else:
            # Comportamentul normal de explorare la inceput
            if variatie_alt_mod > miscare_curenta + 12.0:
                vrem_sa_schimbam = True
            elif timp_expirat and variatie_alt_mod > 8.0:
                vrem_sa_schimbam = True

        # 4. Alegem CEL MAI BUN CANDIDAT din randul celor care chiar se misca
        if vrem_sa_schimbam:
            mod_candidat = cel_mai_activ_alt_mod
            
            if self.sursa_fortei is not None:
                # Daca folosim yolo/mouse, alegem membrul in miscare cel mai apropiat de scripete
                candidati_solizi = {k: v for k, v in alte_membre_in_miscare.items() if v > 8.0}
                if candidati_solizi:
                    distante = {}
                    for mod_key in candidati_solizi:
                        idx_c = self.MAPARE_ARTICULATII[mod_key][2]
                        px_x = landmarks[idx_c.value].x * w
                        px_y = landmarks[idx_c.value].y * h
                        dist = np.sqrt((px_x - self.sursa_fortei[0])**2 + (px_y - self.sursa_fortei[1])**2)
                        distante[mod_key] = dist
                    mod_candidat = min(distante, key=distante.get)

            if mod_candidat != mod_curent_activ:
                self.index_mod = self.lista_moduri.index(mod_candidat)
                self.reset_scor()
                self.timp_ultima_miscare_activa = time.time()

    def identifica_tip_exercitiu(self, unghi_curent, unghi_aux, mod_curent):
        """ 
        Identifica automat exercitiul evaluand combinatia dintre directia miscarii articulatiei active 
        si unghiul postural secundar in momentul intinderii maxime (start).
        """
        # 1. Colectăm date în buffer până la 10 cadre (doar 0.3 secunde)
        if len(self.unghiuri_start_buffer) < 10:
            self.unghiuri_start_buffer.append(unghi_aux)
            return

        # 2. După 10 cadre, calculăm media stabilă pentru start
        if self.unghi_aux_start is None:
            self.unghi_aux_start = sum(self.unghiuri_start_buffer) / len(self.unghiuri_start_buffer)
            # Acum avem un punct de start solid, calculat matematic corect
        
        if self.exercitiu_detectat != "Asteptare miscare...":
            return
            
        if self.unghi_start is None:
            self.unghi_start = unghi_curent
            self.unghi_aux_start = unghi_aux
            return
            
        diferenta = unghi_curent - self.unghi_start
        
        if 'brat' in mod_curent:
            # self.unghi_aux_start = unghiul umarului (Hip-Shoulder-Elbow)
            # Am marit plaja pana la 80 de grade pentru a prinde bratul dus mult la spate 
            # (Bayesian Curls) sau usor in fata (Preacher Curls)
            if self.unghi_aux_start < 80.0:
                if diferenta < -self.prag_detectie:
                    self.exercitiu_detectat = "Flexii Biceps (Orice Unghi)"
                elif diferenta > self.prag_detectie:
                    self.exercitiu_detectat = "Extensii Triceps (Pushdown)"
            elif 80.0 <= self.unghi_aux_start <= 125.0:
                # Pozitie orizontala a bratelor fata de torace (Impins sau Ramat)
                if diferenta < -self.prag_detectie:
                    self.exercitiu_detectat = "Ramat Spate (Rows)"
                elif diferenta > self.prag_detectie:
                    self.exercitiu_detectat = "Impins Piept (Chest Press)"
            elif self.unghi_aux_start > 125.0:
                # Pozitie verticala, deasupra capului
                if diferenta > self.prag_detectie:
                    self.exercitiu_detectat = "Extensii Triceps (Overhead)"
                elif diferenta < -self.prag_detectie:
                    self.exercitiu_detectat = "Tractiuni Spate (Pulldowns)"
                
        elif 'picior' in mod_curent:
            # unghi_curent = unghiul genunchiului (Hip-Knee-Ankle)
            # unghi_aux = unghiul soldului (Shoulder-Hip-Knee)
            # self.unghi_start = unghiul genunchiului initial
            # self.unghi_aux_start = unghiul soldului initial
            
            diferenta_genunchi = unghi_curent - self.unghi_start
            diferenta_sold = unghi_aux - self.unghi_aux_start
            
            # Clasificator optimizat pentru picioare:
            # Daca genunchiul se misca masiv (peste 30 grade), este garantat o miscare compusa
            # precum Genoflexiuni sau Presa, chiar daca soldul pare fix din cauza ocluziei sau erorilor de tracking.
            if np.abs(diferenta_genunchi) > 30.0:
                self.exercitiu_detectat = "Genoflexiuni / Presa Picioare"
            elif np.abs(diferenta_genunchi) > self.prag_detectie:
                # 1. Extensii Cvadriceps (Leg Extension)
                # Genunchiul se intinde dintr-o postura asezata cu soldul stabil
                if diferenta_genunchi > self.prag_detectie and self.unghi_aux_start < 135.0 and np.abs(diferenta_sold) < 5.0:
                    self.exercitiu_detectat = "Extensii Cvadriceps (Leg Ext)"
                
                # 2. Flexii Femurali (Leg Curls)
                # Genunchiul se indoaie (diferenta < 0) cu soldul stabil
                elif diferenta_genunchi < -self.prag_detectie and np.abs(diferenta_sold) < 5.0:
                    self.exercitiu_detectat = "Flexii Femurali (Leg Curls)"
                
                # 3. Genoflexiuni / Presa Picioare (Squats / Leg Press)
                # Co-variatie normala genunchi + sold
                elif np.abs(diferenta_sold) >= 15.0:
                    self.exercitiu_detectat = "Genoflexiuni / Presa Picioare"
                
                # Fallback pentru siguranta
                else:
                    self.exercitiu_detectat = "Genoflexiuni / Presa Picioare"
                    
        elif 'umar' in mod_curent:
            # Pentru umeri, urmarim unghiul umarului (Hip-Shoulder-Elbow) ca articulatie principala
            if diferenta > self.prag_detectie:
                self.exercitiu_detectat = "Ridicari Laterale (Umeri)"
            elif diferenta < -self.prag_detectie:
                self.exercitiu_detectat = "Ramat Vertical (Trapez)"

    def calculeaza_fizica(self, punct_a, pivot, extremitate):
        """ Calculeaza fortele geometrice, unghiurile de rezistenta si tensiunea mecanica transmisa pe parghie. """
        if self.sursa_fortei is not None:
            punct_forta = self.sursa_fortei
        else:
            # Daca nu este selectat niciun scripete, gravitatia actioneaza pur pe axa verticala (Y) in jos
            punct_forta = (extremitate[0], extremitate[1] + 1000)
            
        unghi_art = calculeaza_unghi(punct_a, pivot, extremitate)
        unghi_rez = calculeaza_unghi(pivot, extremitate, punct_forta)
        
        punct_perp = calculeaza_proiectie_perpendiculara(pivot, extremitate, punct_forta)
        
        dist_d = int(np.sqrt((pivot[0] - punct_perp[0])**2 + (pivot[1] - punct_perp[1])**2))
        d_max = max(1, int(np.sqrt((pivot[0] - extremitate[0])**2 + (pivot[1] - extremitate[1])**2)))
        procent_tens = min(100, int((dist_d / d_max) * 100))
        
        return punct_forta, unghi_art, unghi_rez, punct_perp, dist_d, procent_tens

    def evalueaza_hipertrofia(self, extremitate, punct_forta, procent_tensiune, h):
        """ Evalueaza eficienta tensiunii mecanice pe parcursul miscarii (R.O.M.) """
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
            # O nota mai mare daca tensiunea maxima apare in pozitie alungita (stretch-mediated hypertrophy)
            self.nota_numerica = max(1.0, min(10.0, 10.0 - (pozitie_tensiune * 6.0)))
            self.scor_hipertrofie = f"Scor: {self.nota_numerica:.1f} / 10"

    def evalueaza_forma_executie(self, unghi_art, unghi_aux, exercitiu):
        """ Sistem de Form Check care analizeaza tempo-ul negativ, alungirea si trișatul. """
        if exercitiu == "Asteptare miscare...": return

        # Folosim un istoric scurt pentru a vedea in ce directie se misca bratul acum (viteza unghiulara)
        self.istoric_form_unghi.append(unghi_art)
        if len(self.istoric_form_unghi) > 5:
            self.istoric_form_unghi.pop(0)
        if len(self.istoric_form_unghi) < 5: return
        
        delta_unghi = self.istoric_form_unghi[-1] - self.istoric_form_unghi[0]
        
        # Actualizam extremele curente pe parcursul reprizei
        self.unghi_minim_curent = min(self.unghi_minim_curent, unghi_art)
        self.unghi_maxim_curent = max(self.unghi_maxim_curent, unghi_art)
        self.unghi_aux_minim = min(self.unghi_aux_minim, unghi_aux)
        self.unghi_aux_maxim = max(self.unghi_aux_maxim, unghi_aux)

        # Exercitii unde "Impingi" -> Concentric inseamna unghiul articulatiei CRESTE
        EXERCITII_EXTENSIE = [
            "Extensii Triceps (Pushdown)", "Extensii Triceps (Overhead)", 
            "Extensii Cvadriceps (Leg Ext)", "Impins Piept (Chest Press)", 
            "Genoflexiuni / Presa Picioare", "Ridicari Laterale (Umeri)", "Ramat Vertical (Trapez)"
        ]
        
        faza_curenta_reala = self.stadiu_repetare
        
        # Detectam directia doar daca miscarea este curata (peste 2 grade variatie)
        if abs(delta_unghi) > 2.0:
            if exercitiu in EXERCITII_EXTENSIE:
                if delta_unghi > 0: faza_curenta_reala = "concentric"
                else: faza_curenta_reala = "excentric"
            else: # Restul sunt flexii (Tragi) -> Concentric inseamna unghiul SCADE
                if delta_unghi < 0: faza_curenta_reala = "concentric"
                else: faza_curenta_reala = "excentric"

        # LOGICA STATE MACHINE: Tranzitia intre Faze
        if faza_curenta_reala != self.stadiu_repetare:
            
            # Cand incepem partea NEGATIVA, pornim cronometrul
            if faza_curenta_reala == "excentric":
                self.timp_start_faza = time.time()
                self.mesaj_form_check = "Controleaza coborarea..."
                
            # Cand am TERMINAT partea negativa si incepem una noua POZITIVA (Repetare Completa)
            elif faza_curenta_reala == "concentric" and self.stadiu_repetare == "excentric":
                timp_excentric = time.time() - self.timp_start_faza
                rom_curent = self.unghi_maxim_curent - self.unghi_minim_curent
                variatie_aux = self.unghi_aux_maxim - self.unghi_aux_minim
                
                # Evaluarea Formei
                penalizari = []
                scor_rep = 10.0
                
                # 1. TEMPO NEGATIV (Prea rapid = se foloseste gravitatia, nu muschiul)
                if timp_excentric < 1.1:
                    penalizari.append("Negativ prea rapid")
                    scor_rep -= 2.5
                    
                # 2. ALUNGIRE / FULL ROM (Miscare prea scurta)
                if rom_curent < 65.0:
                    penalizari.append("Repetare partiala (Alungire mica)")
                    scor_rep -= 2.0
                    
                # 3. TRIȘAT / MOMENTUM (Se misca o articulatie care ar trebui sa stea fixa)
                if variatie_aux > 20.0:
                    penalizari.append("Trișat (Balans mare)")
                    scor_rep -= 2.5
                    
                if not penalizari:
                    self.mesaj_form_check = "Repetare Perfecta! ✅"
                else:
                    self.mesaj_form_check = " | ".join(penalizari)
                    
                # Inregistram scorul si facem media
                self.repetari_analizate += 1
                self.scor_executie = ((self.scor_executie * (self.repetari_analizate - 1)) + max(1.0, scor_rep)) / self.repetari_analizate
                
                # Resetam senzorii pentru urmatoarea repetare
                self.unghi_minim_curent = 999.0
                self.unghi_maxim_curent = 0.0
                self.unghi_aux_minim = 999.0
                self.unghi_aux_maxim = 0.0

            self.stadiu_repetare = faza_curenta_reala

    # ==============================================================================
    # DESENAREA ELEMENTELOR UI / HUD
    # ==============================================================================
    def deseneaza_grafica_biomecanica(self, image, extremitate, pivot, punct_forta, punct_perp, unghi_art, unghi_rez):
        """ Deseneaza segmentele vectoriale si valorile unghiulare in timp real pe ecran. """
        cv2.putText(image, f"Articulatie: {int(unghi_art)} grd", (pivot[0] + 15, pivot[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Rezistenta: {int(unghi_rez)} grd", (extremitate[0] + 15, extremitate[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(image, extremitate, punct_forta, (0, 0, 255), 3) 
        cv2.line(image, pivot, punct_perp, (0, 255, 0), 4)

    def deseneaza_hud_principal(self, image, procent_tensiune, dist_d, h, w):
        """ Afiseaza casetele de date biomecanice, gradul de tensiune si statistica miscarii active. """
        # Am facut panoul putin mai inalt pentru a incapea Form Check-ul
        deseneaza_panel_transparent(image, (15, 15), (480, 250), (20, 20, 20), 0.7)
        
        status_sistem = "PAUZA" if self.is_paused else "ACTIV"
        culoare_sys = (0, 0, 255) if self.is_paused else (0, 255, 0)
        afiseaza_text_umbrit(image, f"SISTEM: {status_sistem}", (30, 45), 0.7, culoare_sys, 2)
        
        mod_afisaj = self.NUME_MODURI[self.lista_moduri[self.index_mod]]
        afiseaza_text_umbrit(image, f"Membru: {mod_afisaj} [{'Auto' if self.auto_mod else 'Manual'}]", (30, 75), 0.6, (255, 255, 0), 1)
        
        afiseaza_text_umbrit(image, f"Clasificator AI: {self.exercitiu_detectat}", (30, 110), 0.6, (0, 255, 150), 2)
        
        status_yolo = "ON (Cauta)" if self.yolo_activat else "OFF"
        culoare_yolo = (0, 255, 255) if self.yolo_activat else (150, 150, 150)
        afiseaza_text_umbrit(image, f"YOLO AI: {status_yolo}", (30, 140), 0.6, culoare_yolo, 1)
        afiseaza_text_umbrit(image, f"Sursa: {self.tip_forta}", (30, 170), 0.5, (0, 165, 255), 1)
        
        afiseaza_text_umbrit(image, f"Brat Forta (d): {dist_d} px", (30, 200), 0.6, (0, 255, 0), 1)
        
        # NOU: Afisarea sistemului de Form Check
        culoare_form = (0, 255, 0) if "Perfecta" in self.mesaj_form_check else (0, 100, 255)
        if "Astept" in self.mesaj_form_check or "Controleaza" in self.mesaj_form_check: culoare_form = (200, 200, 200)
        afiseaza_text_umbrit(image, f"Forma: {self.mesaj_form_check}", (30, 230), 0.5, culoare_form, 1)

        # Barometru de tensiune mecanica
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
        
        culoare_scor = (0, 255, 0) if self.nota_numerica >= 8 else ((0, 255, 255) if self.nota_numerica >= 5 else (0, 0, 255))
        if "calibreaza" in self.scor_hipertrofie: culoare_scor = (200, 200, 200)
        afiseaza_text_umbrit(image, "Evaluare Profil:", (30, 650), 0.6, (255, 255, 255), 1)
        afiseaza_text_umbrit(image, self.scor_hipertrofie, (30, 680), 0.8, culoare_scor, 2)

        # Panoul de scurtaturi tastatura
        deseneaza_panel_transparent(image, (w - 380, h - 130), (w - 15, h - 15), (20, 20, 20), 0.5)
        afiseaza_text_umbrit(image, "[M] Schimba Mod  | [A] Auto-Membru", (w - 365, h - 105), 0.5, (200, 200, 200), 1)
        afiseaza_text_umbrit(image, "[O] YOLO Camera  | [P] Pauza video", (w - 365, h - 80), 0.5, (200, 200, 200), 1)
        afiseaza_text_umbrit(image, "[Click] Sursa Ft | [Click Dr] Sterge", (w - 365, h - 55), 0.5, (200, 200, 200), 1)
        afiseaza_text_umbrit(image, "[E] ECRAN EVALUARE RAPORT FINAL", (w - 365, h - 30), 0.6, (0, 255, 255), 2)

    def deseneaza_ecran_evaluare(self, image):
        """ Deseneaza raportul analitic final cand utilizatorul apasa tasta 'E'. """
        h, w = image.shape[:2]
        deseneaza_panel_transparent(image, (0, 0), (w, h), (10, 10, 15), 0.85)
        
        caseta_w, caseta_h = 600, 480 # Caseta marita
        sx, sy = (w - caseta_w) // 2, (h - caseta_h) // 2
        cv2.rectangle(image, (sx, sy), (sx + caseta_w, sy + caseta_h), (30, 30, 35), -1)
        cv2.rectangle(image, (sx, sy), (sx + caseta_w, sy + caseta_h), (0, 165, 255), 2)
        
        afiseaza_text_umbrit(image, "RAPORT BIOMECANIC EXERCITIU", (sx + 80, sy + 60), 0.9, (255, 255, 255), 3)
        cv2.line(image, (sx + 50, sy + 80), (sx + caseta_w - 50, sy + 80), (100, 100, 100), 2)
        
        # --- SECTIUNEA 1: PROFIL APARAT ---
        clr_scor = (0, 255, 0) if self.nota_numerica >= 8 else ((0, 255, 255) if self.nota_numerica >= 5 else (0, 0, 255))
        verdict = "Aparat: OPTIM (Hipertrofie Maxima)" if self.nota_numerica >= 8 else ("Aparat: MODERAT (Tensiune acceptabila)" if self.nota_numerica >= 5 else "Aparat: SUB-OPTIM (Tensiune scazuta)")
        
        if "calibreaza" in self.scor_hipertrofie or (self.dist_maxima_rom - self.dist_minima_rom) <= 50.0: 
            clr_scor = (150, 150, 150)
            verdict = "DATE INSUFICIENTE (Executa 1-2 repetiții complete)"
            
        afiseaza_text_umbrit(image, f"SCOR PROFIL APARAT: {self.nota_numerica:.1f} / 10", (sx + 80, sy + 140), 0.9, clr_scor, 3)
        afiseaza_text_umbrit(image, verdict, (sx + 80, sy + 175), 0.6, clr_scor, 2)
        
        # --- SECTIUNEA 2: EXECUTIE SPORTIV ---
        clr_exec = (0, 255, 0) if self.scor_executie >= 8 else ((0, 255, 255) if self.scor_executie >= 5 else (0, 0, 255))
        verdict_exec = "Forma: PERFECTA" if self.scor_executie >= 8 else ("Forma: CU GRESELI (Vezi avertismente)" if self.scor_executie >= 5 else "Forma: SLABA (Risc de accidentare / Ineficient)")
        
        afiseaza_text_umbrit(image, f"SCOR FORMA EXECUTIE: {self.scor_executie:.1f} / 10", (sx + 80, sy + 230), 0.9, clr_exec, 3)
        afiseaza_text_umbrit(image, verdict_exec, (sx + 80, sy + 265), 0.6, clr_exec, 2)
        
        # --- SECTIUNEA 3: RECOMANDARI ---
        cv2.line(image, (sx + 50, sy + 300), (sx + caseta_w - 50, sy + 300), (100, 100, 100), 1)
        
        if self.nota_numerica >= 8.0 and not ("calibreaza" in self.scor_hipertrofie):
            afiseaza_text_umbrit(image, "SFAT DE ANTRENAMENT:", (sx + 50, sy + 340), 0.6, (0, 200, 255), 2)
            afiseaza_text_umbrit(image, "Aparatul e excelent! Accentueaza intinderea activa pe negativ.", (sx + 50, sy + 365), 0.55, (0, 200, 255), 1)
        elif self.nota_numerica < 8.0 and not ("calibreaza" in self.scor_hipertrofie):
            afiseaza_text_umbrit(image, "RECOMANDARE OPTIMIZARE:", (sx + 50, sy + 340), 0.6, (0, 100, 255), 2)
            afiseaza_text_umbrit(image, "Tensiunea scade cand muschiul e intins. Schimba pozitia scripetelui.", (sx + 50, sy + 365), 0.55, (0, 200, 255), 1)
            
        afiseaza_text_umbrit(image, "Apasa 'E' pentru a reveni la analiza in timp real", (sx + 120, sy + 440), 0.55, (150, 150, 150), 1)

    # ==============================================================================
    # BUCLA REALA A PROCESARII VIDEO
    # ==============================================================================
    def ruleaza(self, sursa_video):
        """ Porneste capatarea imaginilor din camera sau fisier si ruleaza algoritmul biomecanic. """
        cap = cv2.VideoCapture(sursa_video)
        cv2.namedWindow('Analiza Biomecanica AI')
        cv2.setMouseCallback('Analiza Biomecanica AI', self.callback_mouse)
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                if not self.is_paused and not self.arata_ecran_final:
                    ret, frame_read = cap.read()
                    if not ret:
                        if sursa_video != 0: 
                            # Looping automat pentru videoclipuri
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.istoric_miscari = {k: [] for k in self.MAPARE_ARTICULATII}
                            continue
                        else: break
                        
                    if sursa_video == 0: frame_read = cv2.flip(frame_read, 1)
                    frame_read = redimensioneaza_cadru(frame_read, inaltime_tinta=720)
                    
                frame = frame_read.copy() 
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- AICI ESTE FIX-UL PENTRU RESETAREA YOLO ---
                # Salvam starea memoriei INAINTE ca YOLO sa ruleze pe cadrul actual
                avea_sursa_inainte = self.sursa_fortei is not None
                
                yolo_gasit_acum, nume_obj, box_coords = self.detecteaza_sursa_yolo(frame, image_rgb)
                
                # Daca YOLO a gasit un scripete NOU (sursa inainte era vida) si rulam un clip
                if yolo_gasit_acum and not avea_sursa_inainte and sursa_video != 0:
                    if not self.calibrare_initiala_yolo_facuta:
                        # Resetam videoclipul la inceput pentru calibrarea miscarii DOAR prima data
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.reset_scor()
                        self.istoric_miscari = {k: [] for k in self.MAPARE_ARTICULATII}
                        self.calibrare_initiala_yolo_facuta = True
                        continue # Sarim peste acest cadru, incepem proaspat
                # ------------------------------------------------
                
                if yolo_gasit_acum:
                    x1, y1, x2, y2 = box_coords
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    titlu = f"Scripete ({nume_obj})" if self.model_is_custom else f"Obiect ({nume_obj})"
                    afiseaza_text_umbrit(image_rgb, titlu, (x1, y1-10), 0.5, (0, 165, 255), 1)
                
                elif self.yolo_activat and self.sursa_fortei is not None:
                    cv2.circle(image_rgb, self.sursa_fortei, 15, (0, 100, 255), 2)
                    afiseaza_text_umbrit(image_rgb, "Memorie", (self.sursa_fortei[0]-35, self.sursa_fortei[1]-20), 0.4, (0, 100, 255), 1)

                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                procent_tens, dist_d = 0, 0
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = image_bgr.shape
                    
                    if not self.is_paused and self.auto_mod and not self.arata_ecran_final:
                        self.identifica_membru_activ(landmarks, w, h)
                        
                    mod_curent = self.lista_moduri[self.index_mod]
                    if mod_curent != self.mod_precedent:
                        self.reset_scor()
                        
                        # --- NOU: Resetam videoclipul daca membrul activ se schimba (dupa initializare) ---
                        if self.mod_precedent is not None and sursa_video != 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.istoric_miscari = {k: [] for k in self.MAPARE_ARTICULATII}
                            self.mod_precedent = mod_curent
                            continue # <--- ADAUGAT: Iesim imediat din iteratia curenta pentru a nu inregistra un unghi fals!
                        # ----------------------------------------------------------------------------------
                        self.mod_precedent = mod_curent

                    is_left = '_s' in mod_curent
                    
                    # Obtinem punctele esentiale dinamice ale partii corpului urmarite (Stanga vs Dreapta)
                    if is_left:
                        lm_hip = self.mp_pose.PoseLandmark.LEFT_HIP
                        lm_shoulder = self.mp_pose.PoseLandmark.LEFT_SHOULDER
                        lm_elbow = self.mp_pose.PoseLandmark.LEFT_ELBOW
                        lm_wrist = self.mp_pose.PoseLandmark.LEFT_WRIST
                        lm_knee = self.mp_pose.PoseLandmark.LEFT_KNEE
                        lm_ankle = self.mp_pose.PoseLandmark.LEFT_ANKLE
                    else:
                        lm_hip = self.mp_pose.PoseLandmark.RIGHT_HIP
                        lm_shoulder = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
                        lm_elbow = self.mp_pose.PoseLandmark.RIGHT_ELBOW
                        lm_wrist = self.mp_pose.PoseLandmark.RIGHT_WRIST
                        lm_knee = self.mp_pose.PoseLandmark.RIGHT_KNEE
                        lm_ankle = self.mp_pose.PoseLandmark.RIGHT_ANKLE

                    pt_hip = tuple(np.multiply([landmarks[lm_hip.value].x, landmarks[lm_hip.value].y], [w, h]).astype(int))
                    pt_shoulder = tuple(np.multiply([landmarks[lm_shoulder.value].x, landmarks[lm_shoulder.value].y], [w, h]).astype(int))
                    pt_elbow = tuple(np.multiply([landmarks[lm_elbow.value].x, landmarks[lm_elbow.value].y], [w, h]).astype(int))
                    pt_wrist = tuple(np.multiply([landmarks[lm_wrist.value].x, landmarks[lm_wrist.value].y], [w, h]).astype(int))
                    pt_knee = tuple(np.multiply([landmarks[lm_knee.value].x, landmarks[lm_knee.value].y], [w, h]).astype(int))
                    pt_ankle = tuple(np.multiply([landmarks[lm_ankle.value].x, landmarks[lm_ankle.value].y], [w, h]).astype(int))

                    # Maparea reperelor biomecanice in functie de modul activ ales
                    idx_a, idx_b, idx_c = self.MAPARE_ARTICULATII[mod_curent]
                    pt_a = tuple(np.multiply([landmarks[idx_a.value].x, landmarks[idx_a.value].y], [w, h]).astype(int))
                    pivot = tuple(np.multiply([landmarks[idx_b.value].x, landmarks[idx_b.value].y], [w, h]).astype(int))
                    extrem = tuple(np.multiply([landmarks[idx_c.value].x, landmarks[idx_c.value].y], [w, h]).astype(int))

                    # Calculeaza fizica de baza a segmentului curent
                    punct_forta, unghi_art, unghi_rez, punct_perp, dist_d, procent_tens = self.calculeaza_fizica(pt_a, pivot, extrem)
                    
                    if self.sursa_fortei is not None:
                        self.tip_forta = f"Aparat ({nume_obj})" if yolo_gasit_acum else "Aparat (Memorie)"
                        if not self.yolo_activat: cv2.circle(image_bgr, self.sursa_fortei, 10, (0, 165, 255), 2)
                    else:
                        self.tip_forta = "Gravitatie (Astept scripete...)" if self.yolo_activat else "Gravitatie"

                    # Aplicarea logicii deterministe pe baza posturii
                    if not self.is_paused and not self.arata_ecran_final:
                        # Calculam unghiul postural secundar necesar pentru clasificator
                        if 'brat' in mod_curent:
                            unghi_aux = calculeaza_unghi(pt_hip, pt_shoulder, pt_elbow)
                        elif 'picior' in mod_curent:
                            unghi_aux = calculeaza_unghi(pt_shoulder, pt_hip, pt_knee)
                        else:  # Modul 'umar'
                            unghi_aux = calculeaza_unghi(pt_shoulder, pt_elbow, pt_wrist)
                            
                        self.identifica_tip_exercitiu(unghi_art, unghi_aux, mod_curent)
                        self.evalueaza_hipertrofia(extrem, punct_forta, procent_tens, h)
                        
                        # AICI SE APELEAZA NOUL MODUL DE FORM CHECK
                        self.evalueaza_forma_executie(unghi_art, unghi_aux, self.exercitiu_detectat)

                    if not self.arata_ecran_final:
                        self.deseneaza_grafica_biomecanica(image_bgr, extrem, pivot, punct_forta, punct_perp, unghi_art, unghi_rez)
                        self.deseneaza_hud_principal(image_bgr, procent_tens, dist_d, h, w)
                
                except Exception:
                    pass 

                if not self.arata_ecran_final:
                    self.mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4), 
                                            self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=2))               

                if self.arata_ecran_final:
                    self.deseneaza_ecran_evaluare(image_bgr)

                cv2.imshow('Analiza Biomecanica AI', image_bgr)

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
                        if not self.yolo_activat: 
                            self.sursa_fortei = None 
                            self.ultima_pozitie_yolo = None  
                        self.reset_scor() 
                    else:
                        print("Libraria ultralytics nu este instalata!")

        cap.release()
        cv2.destroyAllWindows()


# ==============================================================================
# INTERFATA GRAFICA (MENIUL PRINCIPAL - LAUNCHER)
# ==============================================================================
class InterfataPrincipala(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("AI Fitness Biomechanics")
        self.geometry("520x480")
        self.resizable(False, False)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.lbl_titlu = ctk.CTkLabel(self, text="AI BIOMECHANICS", font=ctk.CTkFont(size=28, weight="bold"))
        self.lbl_titlu.pack(pady=(40, 5))
        
        self.lbl_subtitlu = ctk.CTkLabel(self, text="Analiza Tensiunii Mecanice si Hipertrofiei", font=ctk.CTkFont(size=14), text_color="gray")
        self.lbl_subtitlu.pack(pady=(0, 30))

        # Card info cu instructiuni pentru detecția deterministă eficientă
        self.info_card = ctk.CTkFrame(self, fg_color="#1E1E24", border_width=1, border_color="#2D2D34")
        self.info_card.pack(pady=(0, 25), padx=40, fill="x")
        
        self.lbl_info = ctk.CTkLabel(
            self.info_card, 
            text="💡 RECOMANDARE DETECȚIE EXERCIȚII:\nÎncepeți clipul având mușchiul complet întins\npentru calibrarea automată corectă.",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="#F1C40F",
            justify="center"
        )
        self.lbl_info.pack(pady=12, padx=15)
        
        self.btn_camera = ctk.CTkButton(self, text="📹 Folosește Camera Web", height=50, font=ctk.CTkFont(size=15, weight="bold"), command=self.porneste_camera)
        self.btn_camera.pack(pady=10, padx=50, fill="x")
        
        self.btn_video = ctk.CTkButton(self, text="📁 Încarcă Videoclip", height=50, font=ctk.CTkFont(size=15, weight="bold"), fg_color="#E67E22", hover_color="#D35400", command=self.porneste_video)
        self.btn_video.pack(pady=10, padx=50, fill="x")
        
        self.btn_iesire = ctk.CTkButton(self, text="Ieșire", height=40, font=ctk.CTkFont(size=14), fg_color="transparent", border_width=1, text_color="gray", command=self.destroy)
        self.btn_iesire.pack(pady=(20, 10), padx=100, fill="x")
        
    def porneste_camera(self):
        """ Ascunde meniul principal si porneste procesarea camerei web. """
        self.withdraw() 
        aplicatie = AnalizorBiomecanic()
        aplicatie.ruleaza(0) 
        self.deiconify() 
        
    def porneste_video(self):
        """ Deschide dialogul de selectie si porneste procesarea clipului ales. """
        cale_fisier = filedialog.askopenfilename(title="Selecteaza videoclip", filetypes=[("Media", "*.mp4;*.avi;*.mov;*.gif")])
        if cale_fisier:
            self.withdraw()
            aplicatie = AnalizorBiomecanic()
            aplicatie.ruleaza(cale_fisier)
            self.deiconify()

if __name__ == "__main__":
    app_gui = InterfataPrincipala()
    app_gui.mainloop()