import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.spatial import KDTree #pour améliorer la vitesse
import traceback
import os
import random
def redresser_et_rogner_grille(img_hd):
    h, w = img_hd.shape[:2]
    
    gris = cv2.cvtColor(img_hd, cv2.COLOR_BGR2GRAY)
    #passer dans l'espace binaire   
    binary = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 21, 10)
    
    
    #structure horizontale
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 40, 1))
    #struct verticale
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 40))
    #ligne horizontale
    lignes_h = cv2.dilate(cv2.erode(binary, kernel_h), kernel_h)
    #structure verticale
    lignes_v = cv2.dilate(cv2.erode(binary, kernel_v), kernel_v)
    #ne garder que les endroits où les lignes se croisent
    intersections = cv2.bitwise_and(lignes_h, lignes_v)
    #dilater les points pour que la densité soit bien detectable
    grille_points = cv2.dilate(intersections, np.ones((5, 5), np.uint8), iterations=1)
    #cv2.imwrite("image_apres_extraction/lignes_horizontales.png", lignes_h)
    #cv2.imwrite("image_apres_extraction/lignes_verticales.png", lignes_v)
    #cv2.imwrite("image_apres_extraction/intersections.png", grille_points)
    
    
    #analyse densité verticale sur les intersections
    #ecraser nuage de point sur la gauche pour capter densité
    densite_v = np.sum(grille_points, axis=1) / 255
    #trouver les lignes qui contiennent plus de 2 points d'intersection
    indices_y = np.where(densite_v > 2)[0] # Seuil très bas car ce ne sont que des points
    if len(indices_y) > 0:
        #y du haut de notre grille
        y_top = indices_y[0]
        #y du bas de notre grille
        y_bottom = indices_y[-1]
        # Sécurité : on redonne 5-10 pixels de marge pour ne pas couper le trait
        y_top = max(0, y_top - 5)
        y_bottom = min(h, y_bottom + 5)
    else:
        y_top, y_bottom = 0, h
    
    #analyse densité horizontale sur les intersections
    grille_structure = cv2.add(lignes_h, lignes_v)
    #pareil qu'avant mais en écrasant vers le haut
    densite_x = np.sum(grille_structure, axis=0) / 255
    #ne garder que les colonnes qui contiennent au moins 2% de la hauteur de l'image
    indices_x = np.where(densite_x > (h * 0.02))[0]
    x_left, x_right = 0, w
    if len(indices_x) > 0:
        #calculer l'écart entre chaque colonne détecté
        diff = np.diff(indices_x)
        coupures = np.where(diff > 150)[0]
        debuts = np.insert(indices_x[coupures + 1], 0, indices_x[0])
        fins = np.append(indices_x[coupures], indices_x[-1])
        #calculer la largeur de chaque objet trouvé et garder le plus grand
        idx_max = np.argmax(fins - debuts)
        x_left, x_right = debuts[idx_max], fins[idx_max]
    
    #trouver l'angle de la grille
    #trouver l'inclinaison grâce aux lignes verticales
    contours, _ = cv2.findContours(lignes_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle = 0
    if contours:
        #la plus grande ligne verticale
        cnt = max(contours, key=cv2.contourArea)
        #dessiner un rectangle autour de cette ligne et prendre l'angle d'inclinaison
        res_angle = cv2.minAreaRect(cnt)[2]
        angle = res_angle + 90 if res_angle < -45 else res_angle
    
    
    #faire une matrice de rotation qui contient l'angle opposé à la grille
    mat_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    #appliquer matrice sur grille de base pour obtenir grille droite
    img_rot = cv2.warpAffine(img_hd, mat_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
    #crop final
    img_finale = img_rot[y_top:y_bottom, x_left:x_right]
    print(f"Grille isolée (Intersections) : {img_finale.shape[1]}x{img_finale.shape[0]} pixels")
    return img_finale


#chemin_image = "image/HPSC0108.tif"
#image_source = cv2.imread(chemin_image)
#image_resultat = redresser_et_rogner_grille(image_source)
#cv2.imwrite("test_grille_isolee.png", image_resultat)
#print("reulstat crop grille sauvegardé")

def extraction_reconstruction_test1(chemin_img):
    # 1. Chargement et Redressement
    img_hd = cv2.imread(chemin_img)
    if img_hd is None:
        print(f"Erreur : Impossible de charger {chemin_img}")
        return None
        
    # On utilise ta fonction de redressement habituelle
    img = redresser_et_rogner_grille(img_hd)
    
    # 2. Espace colorimétrique et Amélioration
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    # CLAHE pour booster les courbes pâles avant le seuillage
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    v_boosted = clahe.apply(v_channel)
    
    # 3. Seuillage Adaptatif (indispensable pour les courbes fantômes)
    masque_adaptatif = cv2.adaptiveThreshold(
        v_boosted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 51, 15
    )

    # 4. Analyse de la Teinte Dominante (avec protection structurelle)
    # On définit deux zones : une "pâle" (courbe) et une "saturée" (stylo)
    masque_pale = cv2.inRange(hsv, (75, 5, 5), (160, 60, 255))
    masque_sat = cv2.inRange(hsv, (75, 61, 5), (160, 255, 255))

    # Filtre horizontal long pour isoler la courbe du texte
    kernel_long = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    
    # On cherche s'il existe une ligne longue et pâle
    struct_pale = cv2.morphologyEx(cv2.bitwise_and(masque_adaptatif, masque_pale), cv2.MORPH_OPEN, kernel_long)

    if np.any(struct_pale > 0):
        print("Mode : Courbe pâle détectée.")
        hist_final = cv2.calcHist([h_channel], [0], struct_pale, [180], [0, 180])
    else:
        print("Mode : Courbe standard/sombre.")
        cible_sat = cv2.bitwise_and(masque_adaptatif, masque_sat)
        poids_courbe = cv2.morphologyEx(cible_sat, cv2.MORPH_OPEN, kernel_long)
        hist_final = cv2.calcHist([h_channel], [0], cible_sat + (poids_courbe * 10), [180], [0, 180])

    # Nettoyage de l'histogramme (exclusion des rouges/jaunes)
    hist_final[0:75] = 0
    hist_final[160:180] = 0
    teinte_dominante = np.argmax(hist_final)
    print(f"Teinte détectée : {teinte_dominante}")

    # 5. Création du Masque de Couleur Final
    # Sécurité : si on détecte le stylo (125-135), on élargit pour ne pas rater le bleu pâle
    if 120 <= teinte_dominante <= 135:
        basse_h, haute_h = np.array([85, 5, 5]), np.array([140, 255, 255])
    else:
        ecart = 15
        basse_h = np.array([max(0, teinte_dominante - ecart), 5, 5])
        haute_h = np.array([min(179, teinte_dominante + ecart), 255, 255])

    masque_teinte = cv2.inRange(hsv, basse_h, haute_h)
    masque_bleu = cv2.bitwise_and(masque_adaptatif, masque_teinte)

    # 6. Nettoyage Géométrique Final
    # On soude les pointillés
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    masque_bleu = cv2.morphologyEx(masque_bleu, cv2.MORPH_CLOSE, kernel_close)
    
    # On élimine les petits débris (texte, taches)
    nb_labels, labels, stats, _ = cv2.connectedComponentsWithStats(masque_bleu, connectivity=8)
    masque_final = np.zeros_like(masque_bleu)
    for i in range(1, nb_labels):
        if stats[i, cv2.CC_STAT_AREA] > 150: # Garde uniquement les grandes structures
            masque_final[labels == i] = 255

    # 7. Squelettisation
    squelette = cv2.ximgproc.thinning(masque_final)
    
    # Sauvegardes de debug
    cv2.imwrite("image_apres_extraction/debug_1_brut.png", masque_bleu)
    cv2.imwrite("image_apres_extraction/debug_2_propre.png", masque_final)
    cv2.imwrite("image_apres_extraction/debug_3_squelette.png", squelette)
    
    return squelette

chemin_image = "image/HPSC0178.tif"
if os.path.exists(chemin_image):
    print(f"Lancement de l'analyse pour : {chemin_image}")
    resultat = extraction_reconstruction_test1(chemin_image)
    print("Analyse terminée. Les images sont dans le dossier 'image_apres_extraction'.")
else:
    print(f"Erreur : Le fichier {chemin_image} est introuvable.")