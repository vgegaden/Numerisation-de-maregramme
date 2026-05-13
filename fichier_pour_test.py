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
    # 1. Préparation : On utilise la fonction de grille pour avoir une base propre
    img_hd = cv2.imread(chemin_img)
    img = redresser_et_rogner_grille(img_hd) # Appel de la fonction de redressement
    cv2.imwrite("test_grille_isolee.png", img)
    # 2. Ton process d'analyse HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    v_blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
    ret, masque_adaptatif = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Création du masque pour l'histogramme
    masque_pour_histo = cv2.bitwise_and(masque_adaptatif, cv2.threshold(s_channel, 30, 255, cv2.THRESH_BINARY)[1])
    # 3. Calcul de l'histogramme des teintes
    hist_h = cv2.calcHist([h_channel], [0], masque_pour_histo, [180], [0, 180])
    # Nettoyage de l'histogramme pour ignorer l'orange de la grille
    hist_h[0:10] = 0
    hist_h[170:180] = 0
    teinte_dominante = np.argmax(hist_h)
    print(f"Teinte dominante de l'encre détectée : {teinte_dominante}")
    # 4. Création du masque de tolérance
    ecart = 15
    basse_h = max(0, teinte_dominante - ecart)
    haute_h = min(179, teinte_dominante + ecart)
    basse_h_np = np.array([basse_h], dtype="uint8")
    haute_h_np = np.array([haute_h], dtype="uint8")
    masque_teinte = cv2.inRange(h_channel, basse_h_np, haute_h_np)
    # 5. COMBINAISON FINALE (Sombre + Couleur)
    masque_bleu_propre = cv2.bitwise_and(masque_adaptatif, masque_teinte)
    masque_bleu = masque_bleu_propre
    cv2.imwrite("image_apres_extraction/debug_1_extraction_brute.png", masque_bleu)
    # 6. Tes kernels et ton nettoyage morphologique
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    masque_gras = cv2.dilate(masque_bleu, kernel3, iterations=2)
    masque_plein = cv2.morphologyEx(masque_gras, cv2.MORPH_CLOSE, kernel5)
    flou = cv2.GaussianBlur(masque_plein, (5, 5), 0)
    _, masque_propre = cv2.threshold(flou, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("image_apres_extraction/debug_2_extraction_propre.png", masque_propre)
    # 7. Squelettisation
    squelette = cv2.ximgproc.thinning(masque_propre)
    # 8. Nettoyage après squelettisation (Connected Components)
    nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(squelette, connectivity=8)
    squelette_nettoye = np.zeros_like(squelette)
    for i in range(1, nb_labels):
        aire = stats[i, cv2.CC_STAT_AREA]
        if aire > 20: # Ton seuil de 20 pixels
            squelette_nettoye[labels == i] = 255
    squelette = squelette_nettoye
    
    # Sauvegarde finale du résultat
    cv2.imwrite("image_apres_extraction/debug_3_squelette_final.png", squelette)
    return squelette


chemin_image = "image/HPSC0869.tif"
if os.path.exists(chemin_image):
    print(f"Lancement de l'analyse pour : {chemin_image}")
    resultat = extraction_reconstruction_test1(chemin_image)
    print("Analyse terminée. Les images sont dans le dossier 'image_apres_extraction'.")
else:
    print(f"Erreur : Le fichier {chemin_image} est introuvable.")