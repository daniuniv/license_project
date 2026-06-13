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

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        
        # Daca NU suntem pe pauza, citim un cadru nou
        if not is_paused:
            ret, frame_read = cap.read()
            
            # Daca videoclipul s-a terminat, il reluam de la capat pentru a nu se inchide fereastra
            if not ret:
                if sursa != 0: # Daca este un fisier video/gif
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Resetam la primul cadru
                    continue
                else: # Daca e camera web si s-a deconectat
                    print("Camera a fost deconectata.")
                    break
                
            # Daca folosim camera web (0), oglindim imaginea.
            if sursa == 0:
                frame_read = cv2.flip(frame_read, 1)
            else:
                pass
            
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
            
            umar = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            cot = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            
            incheietura = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            unghi_cot = calculeaza_unghi(umar, cot, incheietura)
            
            cot_px = tuple(np.multiply(cot, [w, h]).astype(int))
            incheietura_px = tuple(np.multiply(incheietura, [w, h]).astype(int))
            umar_px = tuple(np.multiply(umar, [w, h]).astype(int))

            cv2.putText(image, str(int(unghi_cot)) + " grade", 
                        cot_px, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            if sursa_fortei is not None:
                punct_forta = sursa_fortei
                tip_forta = "Sursa: Cablu/Scripete"
                cv2.circle(image, sursa_fortei, 15, (0, 165, 255), -1) 
            else:
                punct_forta = (incheietura_px[0], incheietura_px[1] + 1000)
                tip_forta = "Sursa: Gravitatie"
            
            cv2.line(image, incheietura_px, punct_forta, (0, 0, 255), 2)
            punct_perpendicular = calculeaza_proiectie_perpendiculara(cot_px, incheietura_px, punct_forta)
            cv2.line(image, cot_px, punct_perpendicular, (0, 255, 0), 3)
            
            distanta_d = int(np.sqrt((cot_px[0] - punct_perpendicular[0])**2 + (cot_px[1] - punct_perpendicular[1])**2))
            
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
                cv2.putText(image, "Apasa 'P' sau Space pt Pauza", 
                            (50, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            
        except Exception as e:
            pass 

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
        
        cv2.imshow('Analiza Biomecanica AI', image)

        # Citim tastele de la tastatura
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') or key == ord(' '): # Pauza se activeaza cu P sau Space
            is_paused = not is_paused

cap.release()
cv2.destroyAllWindows()