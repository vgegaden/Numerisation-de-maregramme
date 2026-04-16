#test méthode avec segmentation couleur (extraction)
#avec squelettisation (extraction)
#detection de contours (reconstruction)
#ajout logique j+1
#ajout régression linéaire


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def extraction_reconstruction_test1(chemin_img):
    img = cv2.imread(chemin)
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
    all_points = []
    for i in contours:
        for point in i:
            all_points.append(point[0])
    all_points = np.array(all_points)
    all_points = all_points[all_points[:, 0].argsort()]
    
    x=all_points[:, 0]
    y=all_points[:, 1]

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