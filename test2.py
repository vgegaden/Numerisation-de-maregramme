#test méthode avec segmentation couleur (extraction)
#avec squelettisation (extraction)
#detection de contours (reconstruction)
#ajout logique j+1
#ajout régression linéaire
#ajout logique pour trouver la courbe du jour 1
#ajout kdtree 


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.spatial import KDTree #pour améliorer la vitesse

def extraction_reconstruction_test1(chemin_img):
    img = cv2.imread(chemin_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    #partie segmentation couleurs
    bleu_clair = np.array([0, 0, 0])
    bleu_fonce = np.array([179, 255, 100])
    masque_bleu = cv2.inRange(hsv, bleu_clair, bleu_fonce)

    cv2.imwrite("debug_1_extraction_brute.png", masque_bleu)

    kernel_gros = np.ones((5, 5), np.uint8)
    kernel_fin = np.ones((3, 3), np.uint8)

    masque_propre= cv2.dilate(masque_bleu, kernel_gros, iterations = 1)
    masque_propre = cv2.morphologyEx(masque_propre, cv2.MORPH_CLOSE, kernel_fin)
    masque_propre = cv2.medianBlur(masque_propre, 5)

    cv2.imwrite("debug_2_extraction_propre.png", masque_propre)


    #squelettisation
    squelette = cv2.ximgproc.thinning(masque_propre)
    cv2.imwrite("debug_3_squelette.png", squelette)
    #extraction coord 
    contours, _ = cv2.findContours(squelette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points_list = []
    for i in contours:
        if cv2.arcLength(i, True) > 50:
            for point in i:
                points_list.append(tuple(point[0]))
    
    if not points_list:
        return [0],[0],[0]
    
    points_set = set(points_list)
    x_reconstruit = []
    y_reconstruit = []

    #creer un index spatial pour la recherche rapide
    tree = KDTree(points_list)


    #point le plus a gauche
    current_point = min(points_list, key =lambda p: p[0])
    largeur_image = img.shape[1]

    #logique recherche courbe jour 1
    seuil_iso = 15 #distance recherche voisin
    candidats_depart = []
    for p in points_list : 
        if p[0] > 30: #30px pour ne pas commencer au bord
            #chercher si il y a des voisins dans un rayon de 30 px à gauche
            idx_gauche = tree.query_ball_point(p, 30)
            voisins_gauche = [points_list[i] for i in idx_gauche if points_list[i][0] < p[0] - 2]
            #verifier qu'il y a bien une suite à droite (pour s'assurer que c'est une courbe)
            idx_droite = tree.query_ball_point(p, 30)
            voisins_droite = [points_list[i] for i in idx_droite if points_list[i][0] > p[0] + 2]
            if len(voisins_gauche) == 0 and len(voisins_droite) > 10:
                candidats_depart.append(p)

    if candidats_depart:
        current_point = min(candidats_depart, key =lambda p: p[0])
        print(f"Début automatique détecté à : {current_point}")

    else :
        current_point = min(points_list, key =lambda p: p[0])
        print("Début par defaut (bord gauche)")

    #centre_image = largeur_image // 2 
    #pixels du bord
    #seuil_bord = 15
    #nb de points pour la pente
    #nb_points_reg = 15
    #cpt de sauts j+1
    jour_actuel = 0
    x_final, y_final = [], []

    while True:
        x_curr, y_curr = current_point
        x_final.append(x_curr + (jour_actuel * largeur_image))
        y_final.append(y_curr)
        if current_point in points_set:
            points_set.remove(current_point)

        #logique régression
        pente = 0
        #calculer la pente sur les 20 derniers points
        if len(x_final) > 20:
            p = np.polyfit(x_final[-20:], y_final[-20:], 1)
            pente = p[0]
        
        #predire prochain point avec inertie
        cible_x = x_curr + 3
        cible_y = y_curr + (pente*3)

        #logique j+1
        next_pt = None
        seuil_isolement = 10
        #si proche du bord droit
        if x_curr > largeur_image - 20:
            #on cherche au bord gauche (x=5) à la même hauteur
            dist, idx = tree.query([5, y_curr], k=10) #regarder les 10 plus proches
            candidats = [points_list[i] for i in idx if points_list[i] in points_set]
            if candidats:
                next_pt = min(candidats, key=lambda p: abs(p[1] - y_curr))
                jour_actuel += 1
                print(f"passage au jour {jour_actuel+1}")
            
        if next_pt is None :
            #chercher dans un petit rayon autour de la prédiction
            dist, idx = tree.query([x_curr + 2, y_curr + pente], k=15)
            candidats_suivi = [points_list[i] for i in idx if points_list[i] in points_set and points_list[i][0] > x_curr]
            if candidats_suivi:
                next_pt = min(candidats_suivi, key=lambda p: abs(p[1] - (y_curr + pente)))

        if next_pt:
            current_point = next_pt
        else:
            break

    #x=np.array(x_reconstruit)
    #y=np.array(y_reconstruit)

    #y_smooth = medfilt(y, kernel_size=11)
    
    return np.array(x_final), np.array(y_final), medfilt(y_final, kernel_size=15)

chemin = "image/HPSC0869.tif"

try:
    x_val, y_raw, y_final = extraction_reconstruction_test1(chemin)

    plt.figure(figsize=(12, 6))
    #plt.scatter(x_val, y_raw, s=1, color='gray', alpha=0.5, label='Points bruts (Pixels)')
    plt.plot(x_val, y_final, color='blue', label='Signal lissé (Combo 2)')
    plt.title("Numérisation via Combo 2 (OpenCV + Filtrage + KDTree + suivi de pente)")
    plt.gca().invert_yaxis() 
    plt.savefig("resultat_numerisation.png")
    print("resultat enregistré")

except Exception as e:
    print(f"Erreur : {e}.")