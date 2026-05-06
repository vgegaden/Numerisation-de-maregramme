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
import traceback



def redresser_et_rogner_grille(img_hd):
    #travail sur gris car travail sur formes
    gris = cv2.cvtColor(img_hd, cv2.COLOR_BGR2GRAY)
    #flou pour enlever un peu de bruit
    flou = cv2.GaussianBlur(gris, (9,9), 0)
    #seuil pour s'adapter pour faire ressortir le cadre même si l'expo est inégale
    #plus le chiffre a la fin est grand et plus la detection est severe
    seuil = cv2.adaptiveThreshold(flou, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    #dlater lignes pour souder le tout
    kernel5 = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(seuil, kernel5, iterations=2)

    cv2.imwrite("image_apres_extraction/grille_dilate.png", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("pas de contours détecté")
        return img_hd
    '''
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:5]#5 plus grands
    cadre = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx)==4:
            cadre = c
            break

    if cadre is None:
        cadre = contours [0]
    '''
    cadre = max(contours, key=cv2.contourArea)
    #calculer angle pour remettre dans le bon angle
    rect = cv2.minAreaRect(cadre)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else: angle=-angle

    print (f"angle de rota détecté : {angle:.2f} degrés")

    #effectuer la rota de l'image
    (h, w) = img_hd.shape[:2]
    centre = (w // 2, h // 2)
    matrice_rota = cv2.getRotationMatrix2D(centre, angle, 1.0)
    #interpolation lineaire pour garder la qualité et border_replicate pour éviter les bords noirs
    img_redressee = cv2.warpAffine(img_hd, matrice_rota, (w, h), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_REPLICATE)

    #test
    hsv_final = cv2.cvtColor(img_redressee, cv2.COLOR_BGR2HSV)
    bas_rouge1 = np.array([0, 50, 50])
    haut_rouge1 = np.array([10, 255, 255])
    bas_rouge2 = np.array([170, 50, 50])
    haut_rouge2 = np.array([180, 255, 255])
    masque_rouge1 = cv2.inRange(hsv_final, bas_rouge1, haut_rouge1)
    masque_rouge2 = cv2.inRange(hsv_final, bas_rouge2, haut_rouge2)
    masque_rouge = cv2.add(masque_rouge1, masque_rouge2)
    masque_rouge = cv2.morphologyEx(masque_rouge, cv2.MORPH_CLOSE, kernel5)
    contours_rouges, _ = cv2.findContours(masque_rouge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    #rogner grille sur image finale
    gris_final = cv2.cvtColor(img_redressee, cv2.COLOR_BGR2GRAY)
    _, seuil_final = cv2.threshold(gris_final, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_final, _ = cv2.findContours(seuil_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_rouges:
        cadre_rouge = max(contours_rouges, key = cv2.contourArea)
        x, y, w_g, h_g = cv2.boundingRect(cadre_rouge)
        marge_interieur = int(h * 0.04)
        #slicing final
        #on prend le point de depart auquel on ajoute la hauteur et la largeur
        #pour aller jusqu'au bout de la grille et conserver la bonne resolution
        img_finale = img_redressee[y:y+h_g, x:x+w_g]
        print(f"Grille isolée : {w_g}x{h_g} pixels")
        print(f"Grille avec crop brut : {img_finale.shape[1]}x{img_finale.shape[0]}")
        return img_finale

    return img_redressee


chemin_image = "image/HPSC0869.tif" 
image_source = cv2.imread(chemin_image)
image_resultat = redresser_et_rogner_grille(image_source)
cv2.imwrite("test_grille_isolee.png", image_resultat)
print("reulstat crop grille sauvegardé")

'''
if image_source is None:
    print(f"Erreur : Impossible de charger l'image à {chemin_image}")
else:
    # 1. Exécution de la fonction
    try:
        image_resultat = redresser_et_rogner_grille(image_source)

        # 2. Visualisation avec Matplotlib
        plt.figure(figsize=(20, 10))

        # Image Originale
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
        plt.title(f"Originale\n{image_source.shape[1]}x{image_source.shape[0]}")
        plt.axis('off')

        # Image Redressée et Rognée
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image_resultat, cv2.COLOR_BGR2RGB))
        plt.title(f"Résultat (Redressé & Rogné)\n{image_resultat.shape[1]}x{image_resultat.shape[0]}")
        plt.axis('off')

        plt.tight_layout()
        # 3. Sauvegarde pour inspecter les détails (HD)
        cv2.imwrite("test_grille_isolee.png", image_resultat)
        print("Le résultat a été sauvegardé sous 'test_grille_isolee.png'")

    except Exception as e:
        print(f"Une erreur est survenue durant le test : {e}")
'''
