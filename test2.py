#test méthode avec segmentation couleur (extraction)
#avec squelettisation (extraction)
#detection de contours (reconstruction)
#ajout logique j+1
#ajout régression linéaire
#ajout logique pour trouver la courbe du jour 1


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def extraction_reconstruction_test1(chemin_img):
    img = cv2.imread(chemin_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    #partie segmentation couleurs
    bleu_clair = np.array([0, 0, 0])
    bleu_fonce = np.array([179, 255, 100])
    masque_bleu = cv2.inRange(hsv, bleu_clair, bleu_fonce)

    cv2.imwrite("debug_1_extraction_brute.png", masque_bleu)

    kernel = np.ones((3, 3), np.uint8)
    masque_propre = cv2.morphologyEx(masque_bleu, cv2.MORPH_CLOSE, kernel)
    masque_propre = cv2.medianBlur(masque_propre, 5)

    cv2.imwrite("debug_2_extraction_propre.png", masque_propre)


    #squelettisation
    squelette = cv2.ximgproc.thinning(masque_propre)
    cv2.imwrite("debug_3_squelette.png", squelette)
    #extraction coord 
    contours, _ = cv2.findContours(squelette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points_list = []
    for i in contours:
        for point in i:
            points_list.append(tuple(point[0]))
    points_set = set(points_list)
    if not points_list:
        return [0],[0],[0]
    
    x_reconstruit = []
    y_reconstruit = []

    #point le plus a gauche
    point_gauche = min(points_list, key =lambda p: p[0])
    largeur_image = img.shape[1]

    #logique recherche courbe jour 1
    seuil_iso = 15 #distance recherche voisin
    candidats_depart = []
    for p in points_list : 
        if p[0] > 50:
            voisins_gauche = [v for v in points_list if (p[0] - seuil_iso) < v[0] < p[0] and abs(v[1] - p[1]) < 10]
            voisins_droite = [v for v in points_list if p[0] < v[0] < (p[0] + seuil_iso) and abs(v[1] - p[1]) < 10]
            if not voisins_gauche and len(voisins_droite) > 5 :
                candidats_depart.append(p)

    if candidats_depart:
        point_gauche = min(candidats_depart, key =lambda p: p[0])
        print(f"Début automatique détecté à : {point_gauche}")

    else :
        point_gauche = min(points_list, key =lambda p: p[0])
        print("Début par defaut (bord gauche)")

    centre_image = largeur_image // 2 
    #pixels du bord
    seuil_bord = 15
    #nb de points pour la pente
    nb_points_reg = 15
    #cpt de sauts j+1
    jour_actuel = 0

    while points_set:
        x_curr, y_curr = point_gauche
        x_reconstruit.append(x_curr + (jour_actuel * largeur_image))
        y_reconstruit.append(y_curr)
        if point_gauche in points_set:
            points_set.remove(point_gauche)

        #logique régression
        pente = 0
        if len(x_reconstruit) > nb_points_reg:
            pts_x = np.array(x_reconstruit[-nb_points_reg:])
            pts_y = np.array(y_reconstruit[-nb_points_reg:])
            A = np.vstack([pts_x, np.ones(len(pts_x))]).T
            solution, _, _, _ = np.linalg.lstsq(A, pts_y, rcond=None)
            pente = solution[0]

        #logique j+1
        next_pt = None
        seuil_isolement = 10
        #si proche du bord droit
        if x_curr > largeur_image - seuil_bord:
            candidats_gauche = [p for p in points_set if p[0] < seuil_bord and abs(p[1] - y_curr) < 30]
            if candidats_gauche:
                next_pt = min(candidats_gauche, key=lambda p: abs(p[1] - y_curr))
                jour_actuel += 1
                print(f"saut détecté à y= {y_curr}")
            
        if next_pt is None :
            zone_a_scanner = [p for p in points_set if 0 < (p[0] - x_curr) < 6 and abs(p[1] - (y_curr + pente)) < 15]
            if zone_a_scanner:
                next_pt = min(zone_a_scanner, key=lambda p: abs(p[1] - (y_curr + pente)))

        if next_pt:
            point_gauche = next_pt
        else:
            if points_set:
                point_gauche = min(points_set, key=lambda p: (p[0] - x_curr)**2 + (p[1] - y_curr)**2)
                #si écart entre les 2 points trop grand, on stoppe
                if (point_gauche[0] - x_curr)**2 > 900: break
            else :
                break

    x=np.array(x_reconstruit)
    y=np.array(y_reconstruit)

    #filtrage sequentiel
    y_smooth = medfilt(y, kernel_size=11)
    
    return x, y, y_smooth

chemin = "image/HPSC0869.tif"

try:
    x_val, y_raw, y_final = extraction_reconstruction_test1(chemin)

    plt.figure(figsize=(12, 6))
    plt.scatter(x_val, y_raw, s=1, color='gray', alpha=0.5, label='Points bruts (Pixels)')
    plt.plot(x_val, y_final, color='blue', label='Signal lissé (Combo 2)')
    plt.title("Numérisation via Combo 2 (OpenCV + Filtrage)")
    plt.xlabel("Temps (Pixels X)")
    plt.ylabel("Hauteur (Pixels Y)")
    plt.legend()
    plt.gca().invert_yaxis() 
    plt.savefig("resultat_numerisation.png")
    print("resultat enregistré")

except Exception as e:
    print(f"Erreur : {e}.")