import cv2
import numpy as np
import os

def extraction_courbe_hsv(img):
    """
    Isole la couleur de l'encre, nettoie les débris et squelettise la courbe.
    Cette fonction est totalement indépendante et autonome.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    v_blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
    ret, masque_adaptatif = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masque_pour_histo = cv2.bitwise_and(masque_adaptatif, cv2.threshold(s_channel, 30, 255, cv2.THRESH_BINARY)[1])

    # On calcule l'histogramme des teintes (0 à 179)
    hist_h = cv2.calcHist([h_channel], [0], masque_pour_histo, [180], [0, 180])

    # On trouve la teinte dominante (le pic de l'histogramme)
    # On ignore les teintes très proches de 0 ou 180 (souvent du bruit rouge/orange de la grille)
    hist_h[0:10] = 0
    hist_h[170:180] = 0
    teinte_dominante = np.argmax(hist_h)

    print(f"Teinte dominante de l'encre détectée : {teinte_dominante}")

    # On crée un masque de tolérance autour de cette teinte dominante (+/- 15)
    ecart = 15
    basse_h = max(0, teinte_dominante - ecart)
    haute_h = min(179, teinte_dominante + ecart)
    basse_h_np = np.array([basse_h], dtype="uint8")
    haute_h_np = np.array([haute_h], dtype="uint8")

    # On crée un masque qui ne garde que cette plage de teinte
    masque_teinte = cv2.inRange(h_channel, basse_h_np, haute_h_np)

    # COMBINAISON FINALE
    # On ne garde que les pixels qui sont SOMBRES (Otsu) ET de la BONNE COULEUR
    masque_bleu_propre = cv2.bitwise_and(masque_adaptatif, masque_teinte)
    masque_bleu = masque_bleu_propre

    os.makedirs("image_apres_extraction", exist_ok=True)
    cv2.imwrite("image_apres_extraction/debug_1_extraction_brute.png", masque_bleu)

    kernel5 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    
    masque_gras = cv2.dilate(masque_bleu, kernel3, iterations=2)
    masque_plein = cv2.morphologyEx(masque_gras, cv2.MORPH_CLOSE, kernel5)
    flou = cv2.GaussianBlur(masque_plein, (5, 5), 0)
    _, masque_propre = cv2.threshold(flou, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite("image_apres_extraction/debug_2_extraction_propre.png", masque_propre)

    # Squelettisation
    squelette = cv2.ximgproc.thinning(masque_propre)
    
    # Nettoyage après squelettisation (suppression des objets isolés <= 20 pixels)
    nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(squelette, connectivity=8)
    squelette_nettoye = np.zeros_like(squelette)

    for i in range(1, nb_labels):
        aire = stats[i, cv2.CC_STAT_AREA]
        if aire > 20:
            squelette_nettoye[labels == i] = 255

    squelette = squelette_nettoye

    cv2.imwrite("image_apres_extraction/debug_3_squelette.png", squelette)
    return squelette


# ==========================================================
# USAGE DIRECT ET AUTONOME
# ==========================================================
if __name__ == "__main__":
    
    # Indique ici le chemin de ton marégramme scanné d'origine (image brute)
    chemin_maregramme = "image/HPSC0178.tif" 
    
    if os.path.exists(chemin_maregramme):
        print(f"[LANCEMENT] Traitement direct du marégramme : {chemin_maregramme}")
        
        # 1. On charge l'image brute en mémoire
        image_brute = cv2.imread(chemin_maregramme)
        
        # 2. On lance directement l'extraction de la courbe dessus
        squelette_final = extraction_courbe_hsv(image_brute)
        
        print("\n[SUCCÈS] Fin de l'extraction !")
        print("Les 3 étapes de debug de la courbe ont été générées dans : 'image_apres_extraction/'")
    else:
        print(f"[ERREUR] Le fichier '{chemin_maregramme}' est introuvable. Ajuste le chemin dans le code.")
