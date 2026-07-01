import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import time
import customtkinter as ctk

# ==============================================================================
# 1. CONFIGURARI GLOBALE SI CONSTANTE
# ==============================================================================
class ConfigBiomecanica:
    mp_pose = mp.solutions.pose
    MAPARE_ARTICULATII = {
        'brat_s': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        'brat_d': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        'picior_s': (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        'picior_d': (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        'umar_s': (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        'umar_d': (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW)
    }
    
    NUME_MODURI = {
        'brat_s': 'Brat Stang', 'brat_d': 'Brat Drept', 
        'picior_s': 'Picior Stang', 'picior_d': 'Picior Drept',
        'umar_s': 'Umar Stang', 'umar_d': 'Umar Drept'
    }
    LISTA_MODURI = list(MAPARE_ARTICULATII.keys())

# ==============================================================================
# 2. MODULE UTILITARE (STATIC)
# ==============================================================================
class MathUtils:
    @staticmethod
    def calculeaza_unghi(a, b, c):
        """ Calculeaza unghiul format de 3 puncte (2D sau 3D). """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radiani = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        unghi = np.abs(radiani * 180.0 / np.pi)
        return 360 - unghi if unghi > 180.0 else unghi

    @staticmethod
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

class GraphicsUtils:
    @staticmethod
    def deseneaza_panel_transparent(img, top_left, bottom_right, culoare=(0, 0, 0), alpha=0.6):
        overlay = img.copy()
        cv2.rectangle(overlay, top_left, bottom_right, culoare, -1)
        cv2.rectangle(overlay, top_left, bottom_right, (100, 100, 100), 1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    @staticmethod
    def afiseaza_text_umbrit(img, text, pozitie, font_scale=0.6, culoare=(255, 255, 255), grosime=2):
        x, y = pozitie
        cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), grosime + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, culoare, grosime, cv2.LINE_AA)

    @staticmethod
    def redimensioneaza_cadru(frame, inaltime_tinta=720):
        h, w = frame.shape[:2]
        if h == inaltime_tinta: return frame
        raport = inaltime_tinta / float(h)
        return cv2.resize(frame, (int(w * raport), inaltime_tinta))

# ==============================================================================
# 3. COMPONENTE BIOMECANICE SI AI (SRP - Single Responsibility Principle)
# ==============================================================================
class YoloDetector:
    """ Gestioneaza exclusiv detectia sursei de forta prin reteaua neurala YOLO. """
    def __init__(self):
        self.model = None
        self.is_available = False
        self.model_is_custom = False
        self.ultima_pozitie = None
        self.timp_ultima_detectie = 0
        self.timeout = 130.0
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            director_script = os.path.dirname(os.path.abspath(__file__))
            cale_custom = os.path.join(director_script, 'model_aparate.pt')
            if os.path.exists(cale_custom):
                self.model = YOLO(cale_custom)
                self.model_is_custom = True
            else:
                self.model = YOLO('yolov8n.pt') 
                self.model_is_custom = False
            self.is_available = True
        except ImportError:
            self.is_available = False

    def proceseaza_cadru(self, frame_bgr, sursa_curenta, yolo_activat):
        """ Cauta scripetele/aparatul in cadru si aplica un filtru de netezire. """
        obiect_gasit = False
        nume_obiect = ""
        box_coords = None
        noua_sursa = sursa_curenta

        if yolo_activat and self.is_available:
            rezultate = self.model(frame_bgr, verbose=False, conf=0.70)
            for r in rezultate:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    nume = self.model.names[cls]
                    
                    if self.model_is_custom or (nume in ['bottle', 'cup', 'cell phone']):
                        x1, y1, x2, y2 = box.xyxy[0]
                        poz_bruta = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        # Smooth tracking
                        if self.ultima_pozitie:
                            alfa = 0.2  
                            x_smooth = int(self.ultima_pozitie[0] * (1 - alfa) + poz_bruta[0] * alfa)
                            y_smooth = int(self.ultima_pozitie[1] * (1 - alfa) + poz_bruta[1] * alfa)
                            noua_sursa = (x_smooth, y_smooth)
                        else:
                            noua_sursa = poz_bruta
                            
                        self.ultima_pozitie = noua_sursa
                        self.timp_ultima_detectie = time.time()
                        obiect_gasit = True
                        nume_obiect = nume
                        box_coords = (int(x1), int(y1), int(x2), int(y2))
                        break
        
        # Stergem sursa daca a trecut prea mult timp fara detecție
        if not obiect_gasit and yolo_activat:
            if self.timp_ultima_detectie != 0 and (time.time() - self.timp_ultima_detectie < self.timeout):
                pass
            else:
                noua_sursa = None
                self.ultima_pozitie = None
                
        return obiect_gasit, nume_obiect, box_coords, noua_sursa

class FocusManager:
    """ Gestioneaza logica de Lock-In, Timpul de Gratie si comutarea membrului activ. """
    def __init__(self):
        self.reset_istoric()
        self.index_mod = 0
        self.perioada_gratie = 2.0
        self.fereastra_cadre = 25

    def reset_istoric(self):
        self.istoric_miscari = {k: [] for k in ConfigBiomecanica.LISTA_MODURI}
        self.timp_ultima_miscare_activa = time.time()

    def update(self, landmarks, w, h, sursa_fortei, exercitiu_detectat):
        """ Evalueaza miscarile si returneaza True daca membrul activ a fost schimbat. """
        miscari_curente = {}
        visibilitate_buna = {}
        
        for mod_key in ConfigBiomecanica.LISTA_MODURI:
            idx_a, idx_b, idx_c = ConfigBiomecanica.MAPARE_ARTICULATII[mod_key]
            v_a, v_b, v_c = landmarks[idx_a.value].visibility, landmarks[idx_b.value].visibility, landmarks[idx_c.value].visibility
            visibilitate_buna[mod_key] = (v_a > 0.5 and v_b > 0.5 and v_c > 0.5)

            pt_a = [landmarks[idx_a.value].x, landmarks[idx_a.value].y]
            pt_b = [landmarks[idx_b.value].x, landmarks[idx_b.value].y]
            pt_c = [landmarks[idx_c.value].x, landmarks[idx_c.value].y]
            
            unghi_curent = MathUtils.calculeaza_unghi(pt_a, pt_b, pt_c)
            self.istoric_miscari[mod_key].append(unghi_curent)
            
            if len(self.istoric_miscari[mod_key]) > self.fereastra_cadre: 
                self.istoric_miscari[mod_key].pop(0)
            
            if len(self.istoric_miscari[mod_key]) == self.fereastra_cadre:
                if not visibilitate_buna[mod_key]:
                    miscari_curente[mod_key] = 0.0
                else:
                    istoric_sortat = sorted(self.istoric_miscari[mod_key])
                    fara_outliers = istoric_sortat[2:-2] if len(istoric_sortat) > 5 else istoric_sortat
                    miscari_curente[mod_key] = max(fara_outliers) - min(fara_outliers)

        if not miscari_curente: return False

        mod_curent_activ = ConfigBiomecanica.LISTA_MODURI[self.index_mod]
        miscare_curenta = miscari_curente.get(mod_curent_activ, 0.0)
        
        # 1. LOCK-IN: Daca membrul activ se misca peste nivelul de zgomot, actualizam timer-ul (Lock-In)
        if miscare_curenta > 4.0 and visibilitate_buna.get(mod_curent_activ, False):
            self.timp_ultima_miscare_activa = time.time()
            return False # Opreste procesarea pentru alte membre
            
        timp_expirat = (time.time() - self.timp_ultima_miscare_activa) > self.perioada_gratie

        # 2. Analizam CELELALTE membre
        alte_membre_in_miscare = {k: v for k, v in miscari_curente.items() if k != mod_curent_activ and visibilitate_buna.get(k, False)}
        if not alte_membre_in_miscare: return False
            
        cel_mai_activ_alt_mod = max(alte_membre_in_miscare, key=alte_membre_in_miscare.get)
        variatie_alt_mod = alte_membre_in_miscare[cel_mai_activ_alt_mod]
        vrem_sa_schimbam = False
        
        # PREVENTIE FLICKER / RESET FALS:
        if exercitiu_detectat != "Asteptare miscare...":
            if timp_expirat and variatie_alt_mod > 10.0:
                vrem_sa_schimbam = True
        else:
            if variatie_alt_mod > miscare_curenta + 12.0:
                vrem_sa_schimbam = True
            elif timp_expirat and variatie_alt_mod > 8.0:
                vrem_sa_schimbam = True

        # Schimbam focusul (Alegem CEL MAI BUN CANDIDAT)
        if vrem_sa_schimbam:
            mod_candidat = cel_mai_activ_alt_mod
            if sursa_fortei is not None:
                candidati_solizi = {k: v for k, v in alte_membre_in_miscare.items() if v > 8.0}
                if candidati_solizi:
                    distante = {}
                    for mod_key in candidati_solizi:
                        idx_c = ConfigBiomecanica.MAPARE_ARTICULATII[mod_key][2]
                        px_x, px_y = landmarks[idx_c.value].x * w, landmarks[idx_c.value].y * h
                        distante[mod_key] = np.sqrt((px_x - sursa_fortei[0])**2 + (px_y - sursa_fortei[1])**2)
                    mod_candidat = min(distante, key=distante.get)

            if mod_candidat != mod_curent_activ:
                self.index_mod = ConfigBiomecanica.LISTA_MODURI.index(mod_candidat)
                self.timp_ultima_miscare_activa = time.time()
                return True
                
        return False

class ExerciseClassifier:
    """ Determina automat exercitiul evaluand postura in momentul de alungire. """
    def __init__(self):
        self.prag_detectie = 15.0
        self.reset()

    def reset(self):
        self.unghi_start = None
        self.unghi_aux_start = None
        self.exercitiu_detectat = "Asteptare miscare..."
        self.unghiuri_start_buffer = [] 

    def update(self, unghi_curent, unghi_aux, mod_curent):
        # 1. Colectăm date în buffer până la 10 cadre (doar 0.3 secunde)
        if len(self.unghiuri_start_buffer) < 10:
            self.unghiuri_start_buffer.append(unghi_aux)
            return
            
        # 2. După 10 cadre, calculăm media stabilă pentru start
        if self.unghi_aux_start is None:
            self.unghi_aux_start = sum(self.unghiuri_start_buffer) / len(self.unghiuri_start_buffer)
            
        if self.exercitiu_detectat != "Asteptare miscare...":
            return
            
        if self.unghi_start is None:
            self.unghi_start = unghi_curent
            self.unghi_aux_start = unghi_aux
            return
            
        diferenta = unghi_curent - self.unghi_start
        
        if 'brat' in mod_curent:
            if self.unghi_aux_start < 80.0:
                if diferenta < -self.prag_detectie: self.exercitiu_detectat = "Flexii Biceps (Orice Unghi)"
                elif diferenta > self.prag_detectie: self.exercitiu_detectat = "Extensii Triceps (Pushdown)"
            elif 80.0 <= self.unghi_aux_start <= 125.0:
                if diferenta < -self.prag_detectie: self.exercitiu_detectat = "Ramat Spate (Rows)"
                elif diferenta > self.prag_detectie: self.exercitiu_detectat = "Impins Piept (Chest Press)"
            elif self.unghi_aux_start > 125.0:
                if diferenta > self.prag_detectie: self.exercitiu_detectat = "Extensii Triceps (Overhead)"
                elif diferenta < -self.prag_detectie: self.exercitiu_detectat = "Tractiuni Spate (Pulldowns)"
                
        elif 'picior' in mod_curent:
            diferenta_genunchi = unghi_curent - self.unghi_start
            diferenta_sold = unghi_aux - self.unghi_aux_start
            
            if np.abs(diferenta_genunchi) > 30.0:
                self.exercitiu_detectat = "Genoflexiuni / Presa Picioare"
            elif np.abs(diferenta_genunchi) > self.prag_detectie:
                if diferenta_genunchi > self.prag_detectie and self.unghi_aux_start < 135.0 and np.abs(diferenta_sold) < 5.0:
                    self.exercitiu_detectat = "Extensii Cvadriceps (Leg Ext)"
                elif diferenta_genunchi < -self.prag_detectie and np.abs(diferenta_sold) < 5.0:
                    self.exercitiu_detectat = "Flexii Femurali (Leg Curls)"
                elif np.abs(diferenta_sold) >= 15.0:
                    self.exercitiu_detectat = "Genoflexiuni / Presa Picioare"
                else:
                    self.exercitiu_detectat = "Genoflexiuni / Presa Picioare"
                    
        elif 'umar' in mod_curent:
            if diferenta > self.prag_detectie: self.exercitiu_detectat = "Ridicari Laterale (Umeri)"
            elif diferenta < -self.prag_detectie: self.exercitiu_detectat = "Ramat Vertical (Trapez)"

class HypertrophyEvaluator:
    """ Calculeaza eficienta profilului de rezistenta. In modul aparat, foloseste matricea de forta interna. """
    def __init__(self):
        self.reset()

    def reset(self):
        self.dist_minima_rom = 10000.0
        self.dist_maxima_rom = 0.0
        self.dist_la_tensiune_max = 0.0
        self.tensiune_maxima_inregistrata = 0.0
        self.scor_hipertrofie = "Se calibreaza..."
        self.nota_numerica = 0.0

    def update(self, extremitate, punct_forta, procent_tensiune, h, are_sursa_fortei, mod_aparat=False, exercitiu="Asteptare miscare..."):
        if not mod_aparat:
            # --- LOGICA VECHE: Pentru Ganteră / Scripete (Analiză Vectorială) ---
            ancora = punct_forta if are_sursa_fortei else (extremitate[0], h * 2)
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
        else:
            # --- LOGICA NOUĂ: Mod Aparat (Analiza Rezistenței Interne) ---
            if exercitiu == "Asteptare miscare...":
                self.scor_hipertrofie = "Se calibreaza aparatul..."
                self.nota_numerica = 0.0
                return
                
            # Nota de bază pentru un aparat cu tensiune constantă
            scor_baza = 5.0
            modificator = 0.0
            
            # Grupele musculare mai SLABE în alungire (+2 puncte pentru hipertrofie optimă)
            if exercitiu in [
                "Flexii Biceps (Orice Unghi)", 
                "Extensii Triceps (Pushdown)", 
                "Extensii Triceps (Overhead)", 
                "Extensii Cvadriceps (Leg Ext)", 
                "Impins Piept (Chest Press)", 
                "Genoflexiuni / Presa Picioare"
            ]:
                modificator = 2.0
                
            # Grupele musculare mai PUTERNICE în alungire (-1 punct - tensiune suboptima)
            elif exercitiu in [
                "Ramat Spate (Rows)", 
                "Tractiuni Spate (Pulldowns)", 
                "Flexii Femurali (Leg Curls)", 
                "Ridicari Laterale (Umeri)", 
                "Ramat Vertical (Trapez)"
            ]:
                modificator = -1.0
                
            # Calculăm nota finală
            self.nota_numerica = max(1.0, min(10.0, scor_baza + modificator))
            
            # Actualizăm textul afișat pe ecran
            if modificator > 0:
                self.scor_hipertrofie = f"Aparat: {self.nota_numerica:.1f}/10 (Bonus Alungire)"
            elif modificator < 0:
                self.scor_hipertrofie = f"Aparat: {self.nota_numerica:.1f}/10 (Penalizare Alungire)"
            else:
                self.scor_hipertrofie = f"Aparat: {self.nota_numerica:.1f}/10"

class FormEvaluator:
    """ Analizeaza calitatea executiei (tempo, amplitudine, momentum). """
    def __init__(self):
        self.EXERCITII_EXTENSIE = [
            "Extensii Triceps (Pushdown)", "Extensii Triceps (Overhead)", 
            "Extensii Cvadriceps (Leg Ext)", "Impins Piept (Chest Press)", 
            "Genoflexiuni / Presa Picioare", "Ridicari Laterale (Umeri)", "Ramat Vertical (Trapez)"
        ]
        self.reset()

    def reset(self):
        self.scor_executie = 10.0
        self.repetari_analizate = 0
        self.mesaj_form_check = "Astept prima repetare..."
        self.stadiu_repetare = "asteptare" 
        self.unghi_minim_curent = 999.0
        self.unghi_maxim_curent = 0.0
        self.unghi_aux_minim = 999.0
        self.unghi_aux_maxim = 0.0
        self.timp_start_faza = time.time()
        self.istoric_form_unghi = []

    def update(self, unghi_art, unghi_aux, exercitiu):
        if exercitiu == "Asteptare miscare...": return
        
        self.istoric_form_unghi.append(unghi_art)
        if len(self.istoric_form_unghi) > 5: self.istoric_form_unghi.pop(0)
        if len(self.istoric_form_unghi) < 5: return
        
        delta_unghi = self.istoric_form_unghi[-1] - self.istoric_form_unghi[0]
        
        self.unghi_minim_curent = min(self.unghi_minim_curent, unghi_art)
        self.unghi_maxim_curent = max(self.unghi_maxim_curent, unghi_art)
        self.unghi_aux_minim = min(self.unghi_aux_minim, unghi_aux)
        self.unghi_aux_maxim = max(self.unghi_aux_maxim, unghi_aux)

        faza_curenta_reala = self.stadiu_repetare
        if abs(delta_unghi) > 2.0:
            if exercitiu in self.EXERCITII_EXTENSIE:
                faza_curenta_reala = "concentric" if delta_unghi > 0 else "excentric"
            else: 
                faza_curenta_reala = "concentric" if delta_unghi < 0 else "excentric"

        if faza_curenta_reala != self.stadiu_repetare:
            if faza_curenta_reala == "excentric":
                self.timp_start_faza = time.time()
                self.mesaj_form_check = "Controleaza coborarea..."
                
            elif faza_curenta_reala == "concentric" and self.stadiu_repetare == "excentric":
                timp_excentric = time.time() - self.timp_start_faza
                rom_curent = self.unghi_maxim_curent - self.unghi_minim_curent
                variatie_aux = self.unghi_aux_maxim - self.unghi_aux_minim
                
                penalizari = []
                scor_rep = 10.0
                
                if timp_excentric < 1.1:
                    penalizari.append("Negativ prea rapid")
                    scor_rep -= 2.5
                if rom_curent < 65.0:
                    penalizari.append("Repetare partiala (Alungire mica)")
                    scor_rep -= 2.0
                if variatie_aux > 20.0:
                    penalizari.append("Trișat (Balans mare)")
                    scor_rep -= 2.5
                    
                self.mesaj_form_check = "Repetare Perfecta! ✅" if not penalizari else " | ".join(penalizari)
                    
                self.repetari_analizate += 1
                self.scor_executie = ((self.scor_executie * (self.repetari_analizate - 1)) + max(1.0, scor_rep)) / self.repetari_analizate
                
                self.unghi_minim_curent, self.unghi_maxim_curent = 999.0, 0.0
                self.unghi_aux_minim, self.unghi_aux_maxim = 999.0, 0.0

            self.stadiu_repetare = faza_curenta_reala

# ==============================================================================
# 4. ENGINE GRAFIC / RENDERER (Se ocupa strict de desenat)
# ==============================================================================
class UI_Renderer:
    @staticmethod
    def deseneaza_vectori(image, extremitate, pivot, punct_forta, punct_perp, unghi_art, unghi_rez):
        cv2.putText(image, f"Articulatie: {int(unghi_art)} grd", (pivot[0] + 15, pivot[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Rezistenta: {int(unghi_rez)} grd", (extremitate[0] + 15, extremitate[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(image, extremitate, punct_forta, (0, 0, 255), 3) 
        cv2.line(image, pivot, punct_perp, (0, 255, 0), 4)

    @staticmethod
    def deseneaza_hud_principal(image, procent_tensiune, dist_d, h, w, state_dict):
        GraphicsUtils.deseneaza_panel_transparent(image, (15, 15), (480, 250), (20, 20, 20), 0.7)
        
        status_sistem = "PAUZA" if state_dict['is_paused'] else "ACTIV"
        culoare_sys = (0, 0, 255) if state_dict['is_paused'] else (0, 255, 0)
        GraphicsUtils.afiseaza_text_umbrit(image, f"SISTEM: {status_sistem}", (30, 45), 0.7, culoare_sys, 2)
        
        mod_afisaj = ConfigBiomecanica.NUME_MODURI[state_dict['mod_curent']]
        auto_text = 'Auto' if state_dict['auto_mod'] else 'Manual'
        GraphicsUtils.afiseaza_text_umbrit(image, f"Membru: {mod_afisaj} [{auto_text}]", (30, 75), 0.6, (255, 255, 0), 1)
        GraphicsUtils.afiseaza_text_umbrit(image, f"Clasificator AI: {state_dict['exercitiu']}", (30, 110), 0.6, (0, 255, 150), 2)
        
        status_yolo = "ON (Cauta)" if state_dict['yolo_activat'] else "OFF (Aparat Mod)"
        culoare_yolo = (0, 255, 255) if state_dict['yolo_activat'] else (150, 150, 150)
        GraphicsUtils.afiseaza_text_umbrit(image, f"YOLO AI: {status_yolo}", (30, 140), 0.6, culoare_yolo, 1)
        GraphicsUtils.afiseaza_text_umbrit(image, f"Sursa: {state_dict['tip_forta']}", (30, 170), 0.5, (0, 165, 255), 1)
        GraphicsUtils.afiseaza_text_umbrit(image, f"Brat Forta (d): {dist_d} px", (30, 200), 0.6, (0, 255, 0), 1)
        
        mesaj_forma = state_dict['mesaj_form']
        culoare_form = (0, 255, 0) if "Perfecta" in mesaj_forma else (0, 100, 255)
        if "Astept" in mesaj_forma or "Controleaza" in mesaj_forma: culoare_form = (200, 200, 200)
        GraphicsUtils.afiseaza_text_umbrit(image, f"Forma: {mesaj_forma}", (30, 230), 0.5, culoare_form, 1)

        # Barometru
        GraphicsUtils.deseneaza_panel_transparent(image, (15, 420), (280, 700), (20, 20, 20), 0.7)
        bar_x, bar_y, bar_w, bar_h = 50, 460, 40, 150
        r, g = int((procent_tensiune / 100) * 255), int((1 - procent_tensiune / 100) * 255)
        
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(image, (bar_x, bar_y + bar_h - int((procent_tensiune/100)*bar_h)), (bar_x + bar_w, bar_y + bar_h), (0, g, r), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        GraphicsUtils.afiseaza_text_umbrit(image, "TENSIUNE", (bar_x - 15, bar_y - 10), 0.5, (200, 200, 200), 1)
        GraphicsUtils.afiseaza_text_umbrit(image, f"{procent_tensiune}%", (bar_x - 5, bar_y + bar_h + 20), 0.5, (0, g, r), 2)
        
        nota_hip = state_dict['nota_hip']
        culoare_scor = (0, 255, 0) if nota_hip >= 7 else ((0, 255, 255) if nota_hip >= 4 else (0, 0, 255))
        if "calibreaza" in state_dict['scor_text_hip']: culoare_scor = (200, 200, 200)
        GraphicsUtils.afiseaza_text_umbrit(image, "Evaluare Profil:", (30, 650), 0.6, (255, 255, 255), 1)
        GraphicsUtils.afiseaza_text_umbrit(image, state_dict['scor_text_hip'], (30, 680), 0.7, culoare_scor, 2)

        # Controale
        GraphicsUtils.deseneaza_panel_transparent(image, (w - 380, h - 130), (w - 15, h - 15), (20, 20, 20), 0.5)
        GraphicsUtils.afiseaza_text_umbrit(image, "[M] Schimba Mod  | [A] Auto-Membru", (w - 365, h - 105), 0.5, (200, 200, 200), 1)
        GraphicsUtils.afiseaza_text_umbrit(image, "[O] YOLO Camera  | [P] Pauza video", (w - 365, h - 80), 0.5, (200, 200, 200), 1)
        GraphicsUtils.afiseaza_text_umbrit(image, "[Click] Sursa Ft | [Click Dr] Sterge", (w - 365, h - 55), 0.5, (200, 200, 200), 1)
        GraphicsUtils.afiseaza_text_umbrit(image, "[E] ECRAN EVALUARE RAPORT FINAL", (w - 365, h - 30), 0.6, (0, 255, 255), 2)

    @staticmethod
    def deseneaza_ecran_evaluare(image, state_dict):
        h, w = image.shape[:2]
        GraphicsUtils.deseneaza_panel_transparent(image, (0, 0), (w, h), (10, 10, 15), 0.85)
        
        caseta_w, caseta_h = 600, 480
        sx, sy = (w - caseta_w) // 2, (h - caseta_h) // 2
        cv2.rectangle(image, (sx, sy), (sx + caseta_w, sy + caseta_h), (30, 30, 35), -1)
        cv2.rectangle(image, (sx, sy), (sx + caseta_w, sy + caseta_h), (0, 165, 255), 2)
        
        GraphicsUtils.afiseaza_text_umbrit(image, "RAPORT BIOMECANIC EXERCITIU", (sx + 80, sy + 60), 0.9, (255, 255, 255), 3)
        cv2.line(image, (sx + 50, sy + 80), (sx + caseta_w - 50, sy + 80), (100, 100, 100), 2)
        
        # PROFIL APARAT - Adaptat pentru noul sistem de notare
        nota_hip = state_dict['nota_hip']
        clr_scor = (0, 255, 0) if nota_hip >= 7 else ((0, 255, 255) if nota_hip >= 4 else (0, 0, 255))
        verdict = "Echipament: OPTIM (Hipertrofie Maxima)" if nota_hip >= 7 else ("Echipament: MODERAT (Tensiune acceptabila)" if nota_hip >= 4 else "Echipament: SUB-OPTIM (Tensiune scazuta)")
        if "calibreaza" in state_dict['scor_text_hip']: 
            clr_scor = (150, 150, 150)
            verdict = "DATE INSUFICIENTE (Executa 1-2 repetiții complete)"
            
        GraphicsUtils.afiseaza_text_umbrit(image, f"SCOR PROFIL ECHIPAMENT: {nota_hip:.1f} / 10", (sx + 80, sy + 140), 0.9, clr_scor, 3)
        GraphicsUtils.afiseaza_text_umbrit(image, verdict, (sx + 80, sy + 175), 0.6, clr_scor, 2)
        
        # EXECUTIE SPORTIV
        scor_exec = state_dict['scor_executie']
        clr_exec = (0, 255, 0) if scor_exec >= 8 else ((0, 255, 255) if scor_exec >= 5 else (0, 0, 255))
        verdict_exec = "Forma: PERFECTA" if scor_exec >= 8 else ("Forma: CU GRESELI (Vezi avertismente)" if scor_exec >= 5 else "Forma: SLABA (Risc de accidentare / Ineficient)")
        
        GraphicsUtils.afiseaza_text_umbrit(image, f"SCOR FORMA EXECUTIE: {scor_exec:.1f} / 10", (sx + 80, sy + 230), 0.9, clr_exec, 3)
        GraphicsUtils.afiseaza_text_umbrit(image, verdict_exec, (sx + 80, sy + 265), 0.6, clr_exec, 2)
        
        # RECOMANDARI
        cv2.line(image, (sx + 50, sy + 300), (sx + caseta_w - 50, sy + 300), (100, 100, 100), 1)
        if nota_hip >= 7.0 and not ("calibreaza" in state_dict['scor_text_hip']):
            GraphicsUtils.afiseaza_text_umbrit(image, "SFAT DE ANTRENAMENT:", (sx + 50, sy + 340), 0.6, (0, 200, 255), 2)
            GraphicsUtils.afiseaza_text_umbrit(image, "Aparatul e excelent! Accentueaza intinderea activa pe negativ.", (sx + 50, sy + 365), 0.55, (0, 200, 255), 1)
        elif nota_hip < 7.0 and not ("calibreaza" in state_dict['scor_text_hip']):
            GraphicsUtils.afiseaza_text_umbrit(image, "RECOMANDARE OPTIMIZARE:", (sx + 50, sy + 340), 0.6, (0, 100, 255), 2)
            GraphicsUtils.afiseaza_text_umbrit(image, "Aparatul pierde eficienta pe alungire. Incearca alt aparat / unghi.", (sx + 50, sy + 365), 0.55, (0, 200, 255), 1)
            
        GraphicsUtils.afiseaza_text_umbrit(image, "Apasa 'E' pentru a reveni la analiza in timp real", (sx + 120, sy + 440), 0.55, (150, 150, 150), 1)

# ==============================================================================
# 5. CONTROLLER PRINCIPAL (Facade Pattern - Orchestrarea modulelor)
# ==============================================================================
class BiomechanicsAppController:
    def __init__(self, mod_aparat=False):
        # Initializam Sub-sistemele (Compozitie OOP)
        self.yolo = YoloDetector()
        self.focus = FocusManager()
        self.classifier = ExerciseClassifier()
        self.hypertrophy = HypertrophyEvaluator()
        self.form = FormEvaluator()
        self.mp_pose = ConfigBiomecanica.mp_pose
        
        # Stari si Flag-uri Globale
        self.mod_aparat = mod_aparat
        self.is_paused = False
        self.arata_ecran_final = False
        self.auto_mod = True
        
        # Daca suntem in mod_aparat, dezactivam YOLO implicit
        self.yolo_activat = not self.mod_aparat 
        self.sursa_fortei = None
        self.calibrare_initiala_yolo_facuta = False
        self.mod_precedent = None

    def reset_toate_scorurile(self):
        """ Deleaga resetarea catre toate modulele de logica. """
        self.classifier.reset()
        self.hypertrophy.reset()
        self.form.reset()

    def callback_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sursa_fortei = (x, y)
            self.yolo.ultima_pozitie = None
            self.reset_toate_scorurile()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.sursa_fortei = None
            self.yolo.ultima_pozitie = None
            self.reset_toate_scorurile()

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
                            self.focus.reset_istoric()
                            continue
                        else: break
                        
                    if sursa_video == 0: frame_read = cv2.flip(frame_read, 1)
                    frame_read = GraphicsUtils.redimensioneaza_cadru(frame_read, inaltime_tinta=720)
                    
                frame = frame_read.copy() 
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- PROCESARE YOLO ---
                avea_sursa_inainte = self.sursa_fortei is not None
                yolo_gasit_acum, nume_obj, box_coords, self.sursa_fortei = self.yolo.proceseaza_cadru(frame, self.sursa_fortei, self.yolo_activat)
                
                if yolo_gasit_acum and not avea_sursa_inainte and sursa_video != 0:
                    if not self.calibrare_initiala_yolo_facuta:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.reset_toate_scorurile()
                        self.focus.reset_istoric()
                        self.calibrare_initiala_yolo_facuta = True
                        continue 
                
                if yolo_gasit_acum:
                    x1, y1, x2, y2 = box_coords
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    titlu = f"Scripete ({nume_obj})" if self.yolo.model_is_custom else f"Obiect ({nume_obj})"
                    GraphicsUtils.afiseaza_text_umbrit(image_rgb, titlu, (x1, y1-10), 0.5, (0, 165, 255), 1)
                elif self.yolo_activat and self.sursa_fortei is not None:
                    cv2.circle(image_rgb, self.sursa_fortei, 15, (0, 100, 255), 2)
                    GraphicsUtils.afiseaza_text_umbrit(image_rgb, "Memorie", (self.sursa_fortei[0]-35, self.sursa_fortei[1]-20), 0.4, (0, 100, 255), 1)

                # --- PROCESARE POSE ---
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                procent_tens, dist_d, tip_forta = 0, 0, ""
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = image_bgr.shape
                    
                    if not self.is_paused and self.auto_mod and not self.arata_ecran_final:
                        schimbat = self.focus.update(landmarks, w, h, self.sursa_fortei, self.classifier.exercitiu_detectat)
                        mod_curent = ConfigBiomecanica.LISTA_MODURI[self.focus.index_mod]
                        
                        if schimbat or mod_curent != self.mod_precedent:
                            self.reset_toate_scorurile()
                            if self.mod_precedent is not None and sursa_video != 0:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                self.focus.reset_istoric()
                                self.mod_precedent = mod_curent
                                continue 
                            self.mod_precedent = mod_curent

                    mod_curent = ConfigBiomecanica.LISTA_MODURI[self.focus.index_mod]
                    is_left = '_s' in mod_curent
                    
                    lm_hip = self.mp_pose.PoseLandmark.LEFT_HIP if is_left else self.mp_pose.PoseLandmark.RIGHT_HIP
                    lm_shoulder = self.mp_pose.PoseLandmark.LEFT_SHOULDER if is_left else self.mp_pose.PoseLandmark.RIGHT_SHOULDER
                    lm_elbow = self.mp_pose.PoseLandmark.LEFT_ELBOW if is_left else self.mp_pose.PoseLandmark.RIGHT_ELBOW
                    lm_wrist = self.mp_pose.PoseLandmark.LEFT_WRIST if is_left else self.mp_pose.PoseLandmark.RIGHT_WRIST
                    lm_knee = self.mp_pose.PoseLandmark.LEFT_KNEE if is_left else self.mp_pose.PoseLandmark.RIGHT_KNEE
                    lm_ankle = self.mp_pose.PoseLandmark.LEFT_ANKLE if is_left else self.mp_pose.PoseLandmark.RIGHT_ANKLE
                    
                    pt_hip = tuple(np.multiply([landmarks[lm_hip.value].x, landmarks[lm_hip.value].y], [w, h]).astype(int))
                    pt_shoulder = tuple(np.multiply([landmarks[lm_shoulder.value].x, landmarks[lm_shoulder.value].y], [w, h]).astype(int))
                    pt_elbow = tuple(np.multiply([landmarks[lm_elbow.value].x, landmarks[lm_elbow.value].y], [w, h]).astype(int))
                    pt_wrist = tuple(np.multiply([landmarks[lm_wrist.value].x, landmarks[lm_wrist.value].y], [w, h]).astype(int))
                    pt_knee = tuple(np.multiply([landmarks[lm_knee.value].x, landmarks[lm_knee.value].y], [w, h]).astype(int))
                    pt_ankle = tuple(np.multiply([landmarks[lm_ankle.value].x, landmarks[lm_ankle.value].y], [w, h]).astype(int))

                    idx_a, idx_b, idx_c = ConfigBiomecanica.MAPARE_ARTICULATII[mod_curent]
                    pt_a = tuple(np.multiply([landmarks[idx_a.value].x, landmarks[idx_a.value].y], [w, h]).astype(int))
                    pivot = tuple(np.multiply([landmarks[idx_b.value].x, landmarks[idx_b.value].y], [w, h]).astype(int))
                    extrem = tuple(np.multiply([landmarks[idx_c.value].x, landmarks[idx_c.value].y], [w, h]).astype(int))

                    # Calcule Biomecanice Directe
                    punct_forta = self.sursa_fortei if self.sursa_fortei is not None else (extrem[0], extrem[1] + 1000)
                    unghi_art = MathUtils.calculeaza_unghi(pt_a, pivot, extrem)
                    unghi_rez = MathUtils.calculeaza_unghi(pivot, extrem, punct_forta)
                    punct_perp = MathUtils.calculeaza_proiectie_perpendiculara(pivot, extrem, punct_forta)
                    
                    dist_d = int(np.sqrt((pivot[0] - punct_perp[0])**2 + (pivot[1] - punct_perp[1])**2))
                    d_max = max(1, int(np.sqrt((pivot[0] - extrem[0])**2 + (pivot[1] - extrem[1])**2)))
                    procent_tens = min(100, int((dist_d / d_max) * 100))
                    
                    # Suprascriem tensiunea daca e Mod Aparat
                    if self.mod_aparat:
                        procent_tens = 100
                        tip_forta = "Aparat (Rezistenta Int.)"
                    else:
                        if self.sursa_fortei is not None:
                            tip_forta = f"Cablu/Sursa ({nume_obj})" if yolo_gasit_acum else "Cablu (Memorie)"
                            if not self.yolo_activat: cv2.circle(image_bgr, self.sursa_fortei, 10, (0, 165, 255), 2)
                        else:
                            tip_forta = "Gravitatie (Cauta scripete)" if self.yolo_activat else "Gravitatie"

                    # Module logice de Evaluare
                    if not self.is_paused and not self.arata_ecran_final:
                        if 'brat' in mod_curent: unghi_aux = MathUtils.calculeaza_unghi(pt_hip, pt_shoulder, pt_elbow)
                        elif 'picior' in mod_curent: unghi_aux = MathUtils.calculeaza_unghi(pt_shoulder, pt_hip, pt_knee)
                        else: unghi_aux = MathUtils.calculeaza_unghi(pt_shoulder, pt_elbow, pt_wrist)
                            
                        self.classifier.update(unghi_art, unghi_aux, mod_curent)
                        # Trimitem self.mod_aparat si exercitiul pentru analiza interna
                        self.hypertrophy.update(extrem, punct_forta, procent_tens, h, self.sursa_fortei is not None, self.mod_aparat, self.classifier.exercitiu_detectat)
                        self.form.update(unghi_art, unghi_aux, self.classifier.exercitiu_detectat)

                    # Randare Grafica Timp Real
                    if not self.arata_ecran_final:
                        if not self.mod_aparat:
                            UI_Renderer.deseneaza_vectori(image_bgr, extrem, pivot, punct_forta, punct_perp, unghi_art, unghi_rez)
                        stare_curenta = {
                            'is_paused': self.is_paused, 'mod_curent': mod_curent, 'auto_mod': self.auto_mod,
                            'exercitiu': self.classifier.exercitiu_detectat, 'yolo_activat': self.yolo_activat,
                            'tip_forta': tip_forta, 'nota_hip': self.hypertrophy.nota_numerica,
                            'scor_text_hip': self.hypertrophy.scor_hipertrofie, 
                            'scor_executie': self.form.scor_executie, 'mesaj_form': self.form.mesaj_form_check
                        }
                        UI_Renderer.deseneaza_hud_principal(image_bgr, procent_tens, dist_d, h, w, stare_curenta)
                
                except Exception:
                    pass 

                # MediaPipe Overlay
                if not self.arata_ecran_final:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4), 
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=2)
                    )               

                if self.arata_ecran_final:
                    stare_evaluare = {
                        'nota_hip': self.hypertrophy.nota_numerica,
                        'scor_text_hip': self.hypertrophy.scor_hipertrofie,
                        'scor_executie': self.form.scor_executie
                    }
                    UI_Renderer.deseneaza_ecran_evaluare(image_bgr, stare_evaluare)

                cv2.imshow('Analiza Biomecanica AI', image_bgr)

                # Gestionare Evenimente Tastatura
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'): break
                elif key == ord('e'): 
                    self.arata_ecran_final = not self.arata_ecran_final
                    self.is_paused = self.arata_ecran_final
                elif key == ord('p') or key == ord(' '): 
                    if not self.arata_ecran_final: self.is_paused = not self.is_paused
                elif key == ord('m') and not self.arata_ecran_final: 
                    self.auto_mod = False 
                    self.focus.index_mod = (self.focus.index_mod + 1) % len(ConfigBiomecanica.LISTA_MODURI)
                elif key == ord('a') and not self.arata_ecran_final: 
                    self.auto_mod = not self.auto_mod
                    if self.auto_mod: self.focus.reset_istoric()
                elif key == ord('o') and not self.arata_ecran_final and not self.mod_aparat: 
                    if self.yolo.is_available:
                        self.yolo_activat = not self.yolo_activat
                        if not self.yolo_activat: 
                            self.sursa_fortei = None 
                            self.yolo.ultima_pozitie = None  
                        self.reset_toate_scorurile() 
                    else:
                        print("Libraria ultralytics nu este instalata!")

        cap.release()
        cv2.destroyAllWindows()


# ==============================================================================
# 6. INTERFATA GRAFICA DE LANSARE (GUI)
# ==============================================================================
class InterfataPrincipala(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Fitness Biomechanics")
        self.geometry("520x550")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.lbl_titlu = ctk.CTkLabel(self, text="AI BIOMECHANICS", font=ctk.CTkFont(size=28, weight="bold"))
        self.lbl_titlu.pack(pady=(40, 5))
        
        self.lbl_subtitlu = ctk.CTkLabel(self, text="Analiza Tensiunii Mecanice si Hipertrofiei", font=ctk.CTkFont(size=14), text_color="gray")
        self.lbl_subtitlu.pack(pady=(0, 20))

        self.info_card = ctk.CTkFrame(self, fg_color="#1E1E24", border_width=1, border_color="#2D2D34")
        self.info_card.pack(pady=(0, 20), padx=40, fill="x")
        
        self.lbl_info = ctk.CTkLabel(
            self.info_card, 
            text="💡 RECOMANDARE DETECȚIE EXERCIȚII:\nÎncepeți clipul având mușchiul complet întins\npentru calibrarea automată corectă.",
            font=ctk.CTkFont(size=12, slant="italic"), text_color="#F1C40F", justify="center"
        )
        self.lbl_info.pack(pady=12, padx=15)
        
        # NOU: Checkbox pentru Mod Aparat
        self.chk_aparat_var = tk.BooleanVar(value=False)
        self.chk_aparat = ctk.CTkCheckBox(
            self, 
            text="Execut la APARAT (Folosește rezistența internă)", 
            variable=self.chk_aparat_var,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#00FF96"
        )
        self.chk_aparat.pack(pady=(0, 20))
        
        self.btn_camera = ctk.CTkButton(self, text="📹 Folosește Camera Web", height=50, font=ctk.CTkFont(size=15, weight="bold"), command=self.porneste_camera)
        self.btn_camera.pack(pady=10, padx=50, fill="x")
        
        self.btn_video = ctk.CTkButton(self, text="📁 Încarcă Videoclip", height=50, font=ctk.CTkFont(size=15, weight="bold"), fg_color="#E67E22", hover_color="#D35400", command=self.porneste_video)
        self.btn_video.pack(pady=10, padx=50, fill="x")
        
        self.btn_iesire = ctk.CTkButton(self, text="Ieșire", height=40, font=ctk.CTkFont(size=14), fg_color="transparent", border_width=1, text_color="gray", command=self.destroy)
        self.btn_iesire.pack(pady=(15, 10), padx=100, fill="x")
        
    def porneste_camera(self):
        self.withdraw() 
        aplicatie = BiomechanicsAppController(mod_aparat=self.chk_aparat_var.get())
        aplicatie.ruleaza(0) 
        self.deiconify() 
        
    def porneste_video(self):
        cale_fisier = filedialog.askopenfilename(title="Selecteaza videoclip", filetypes=[("Media", "*.mp4;*.avi;*.mov;*.gif")])
        if cale_fisier:
            self.withdraw()
            aplicatie = BiomechanicsAppController(mod_aparat=self.chk_aparat_var.get())
            aplicatie.ruleaza(cale_fisier)
            self.deiconify()

if __name__ == "__main__":
    app_gui = InterfataPrincipala()
    app_gui.mainloop()