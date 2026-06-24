import cv2
import os
import glob

def extrage_cadre_din_folder(folder_video, folder_destinatie, cadre_pe_secunda=2):
    """
    Acest script parcurge un folder intreg de videoclipuri si extrage poze din toate.
    Ideal pentru a crea un set de date diversificat (mai multe aparate/scripeti diferiti).
    """
    
    # Cream folderul unde se vor salva pozele daca nu exista deja
    if not os.path.exists(folder_destinatie):
        os.makedirs(folder_destinatie)
        
    # Cautam toate fisierele video (.mp4, .avi, .mov) din folderul specificat
    fisiere_video = []
    pentru_extensii = ['*.mp4', '*.avi', '*.mov', '*.MP4']
    for extensie in pentru_extensii:
        fisiere_video.extend(glob.glob(os.path.join(folder_video, extensie)))
        
    if not fisiere_video:
        print(f"Eroare: Nu am gasit niciun videoclip in folderul '{folder_video}'.")
        return

    print(f"Am gasit {len(fisiere_video)} videoclip(uri). Incepem procesarea...")
    
    poze_totale_salvate = 0
    
    # Parcurgem fiecare videoclip gasit
    for cale_video in fisiere_video:
        cap = cv2.VideoCapture(cale_video)
        
        if not cap.isOpened():
            print(f"  -> Eroare la deschiderea: {cale_video}")
            continue

        fps_video = int(cap.get(cv2.CAP_PROP_FPS))
        # Prevenim impartirea la zero in caz ca videoclipul e corupt
        if fps_video == 0:
            fps_video = 30 
            
        interval_salvare = max(1, int(fps_video / cadre_pe_secunda))
        
        frame_count = 0
        nume_baza_video = os.path.splitext(os.path.basename(cale_video))[0]
        
        print(f"  -> Extragem din: {nume_baza_video}...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break # S-a terminat acest videoclip
                
            if frame_count % interval_salvare == 0:
                # Numele pozei va contine si numele videoclipului din care provine
                nume_fisier = os.path.join(folder_destinatie, f"scripete_{nume_baza_video}_{poze_totale_salvate:04d}.jpg")
                cv2.imwrite(nume_fisier, frame)
                poze_totale_salvate += 1
                
            frame_count += 1

        cap.release()

    print("="*40)
    print(f"GATA! Au fost extrase in total {poze_totale_salvate} imagini din {len(fisiere_video)} videoclipuri.")
    print(f"Le gasesti in folderul: '{folder_destinatie}'")

# ==========================================
# CUM SE FOLOSESTE SCRIPTUL
# ==========================================
if __name__ == "__main__":
    # 1. Creeaza un folder numit "videoclipurile_mele" langa acest script 
    # si pune acolo videoclipul (sau videoclipurile) tau cu scripeti.
    folder_sursa_videoclipuri = "videoclipurile_mele" 
    
    # (Optional) Cream folderul sursa din cod, ca sa nu dea eroare daca uiti tu
    if not os.path.exists(folder_sursa_videoclipuri):
        os.makedirs(folder_sursa_videoclipuri)
        print(f"Am creat folderul '{folder_sursa_videoclipuri}'. Pune videoclipurile tale acolo si ruleaza din nou scriptul!")
    else:
        # 2. Numele folderului unde vrei sa apara toate pozele extrase
        folder_poze_extrase = "dataset_imagini_scripete" 
        
        # 3. Executam functia (extrage 3 poze pentru fiecare secunda de video)
        extrage_cadre_din_folder(folder_sursa_videoclipuri, folder_poze_extrase, cadre_pe_secunda=3)