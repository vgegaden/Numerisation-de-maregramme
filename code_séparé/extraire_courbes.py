import cv2
import numpy as np
import os

def extraction_courbe_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    v_blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
    ret, masque_adaptatif = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masque_pour_histo = cv2.bitwise_and(masque_adaptatif, cv2.threshold(s_channel, 30, 255, cv2.THRESH_BINARY)[1])

    # On calcule l'histogramme des teintes (0 à 179)
    hist_h = cv2.calcHist([h_channel], [0], masque_pour_histo, [180], [0, 180])

    #On trouve la teinte dominante (le pic de l'histogramme)
    # On ignore les teintes très proches de 0 ou 180 (souvent du bruit rouge/orange de la grille)
    hist_h[0:10] = 0
    hist_h[170:180] = 0
    teinte_dominante = np.argmax(hist_h)

    print(f"Teinte dominante de l'encre détectée : {teinte_dominante}")

    #On crée un masque de tolérance autour de cette teinte dominante (+/- 15)
    ecart = 15
    basse_h = max(0, teinte_dominante - ecart)
    haute_h = min(179, teinte_dominante + ecart)
    basse_h_np = np.array([basse_h], dtype="uint8")
    haute_h_np = np.array([haute_h], dtype="uint8")

    # On crée un masque qui ne garde que cette plage de teinte
    masque_teinte = cv2.inRange(h_channel, basse_h_np, haute_h_np)

    #COMBINAISON FINALE
    # On ne garde que les pixels qui sont SOMBRES (Otsu) ET de la BONNE COULEUR
    masque_bleu_propre = cv2.bitwise_and(masque_adaptatif, masque_teinte)

    # On remplace l'ancien masque pour la suite de ton nettoyage morphologique
    masque_bleu = masque_bleu_propre


    os.makedirs("image_apres_extraction", exist_ok=True)
    cv2.imwrite("image_apres_extraction/debug_1_extraction_brute.png", masque_bleu)

    kernel5 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel7 = np.ones((7, 7), np.uint8)
    kernel4 = np.ones((4, 4), np.uint8)
    masque_gras= cv2.dilate(masque_bleu, kernel3, iterations = 2)
    masque_plein = cv2.morphologyEx(masque_gras, cv2.MORPH_CLOSE, kernel5)
    flou = cv2.GaussianBlur(masque_plein, (5, 5), 0)
    _, masque_propre = cv2.threshold(flou, 127, 255, cv2.THRESH_BINARY)
    #masque_bouche_trou = cv2.morphologyEx(masque_bleu, cv2.MORPH_CLOSE, kernel5)
    
    #masque_sans_bruit = cv2.morphologyEx(masque_bouche_trou, cv2.MORPH_OPEN, kernel5)
    
    #masque_propre = cv2.morphologyEx(masque_propre, cv2.MORPH_OPEN, kernel3)
    #masque_propre = cv2.medianBlur(masque_gras, 5)

    #kernel_bouche_trou = np.ones((5, 5), np.uint8)
    #masque_lisse = cv2.morphologyEx(masque_propre, cv2.MORPH_CLOSE, kernel_bouche_trou)


    cv2.imwrite("image_apres_extraction/debug_2_extraction_propre.png", masque_propre)


    #squelettisation
    squelette = cv2.ximgproc.thinning(masque_propre)
    
    #Nettoyage après squelettisation ---

    # 1. On identifie tous les objets isolés sur le squelette
    # Puisque le squelette est déjà en 0/255, c'est parfait pour la fonction.
    # connectivity=8 est crucial pour ne pas briser la courbe
    nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(squelette, connectivity=8)

    # 2. On crée une image vide pour reconstruire le squelette propre
    squelette_nettoye = np.zeros_like(squelette)

    # L'index 0 est le fond noir, on commence donc à 1 pour les objets
    for i in range(1, nb_labels):
        # On récupère l'aire en pixels de l'objet i
        aire = stats[i, cv2.CC_STAT_AREA]
        
        # Seuil : on supprime tout ce qui est <= 10 pixels
        if aire > 20:
            # On réécrit cet objet dans notre image propre
            squelette_nettoye[labels == i] = 255

    #On morale l'ancien squelette
    squelette = squelette_nettoye
    # ------------------------------------------------

    cv2.imwrite("image_apres_extraction/debug_3_squelette.png", squelette)
    return squelette