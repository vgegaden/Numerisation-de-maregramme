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

    kernel5 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)

    masque_gras= cv2.dilate(masque_bleu, kernel5, iterations = 1)
    masque_propre = cv2.morphologyEx(masque_gras, cv2.MORPH_CLOSE, kernel5)
    #masque_propre = cv2.morphologyEx(masque_propre, cv2.MORPH_OPEN, kernel3)
    masque_propre = cv2.medianBlur(masque_propre, 5)

    #kernel_bouche_trou = np.ones((5, 5), np.uint8)
    #masque_lisse = cv2.morphologyEx(masque_propre, cv2.MORPH_CLOSE, kernel_bouche_trou)


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
    longueur_chaine_min = 25

    debut_recherhce = int(largeur_image*0.6)

    for i in range (0, len(points_list),5) :
        p = points_list[i] 
        if p[0] > debut_recherhce: #50px pour ne pas commencer au bord
            #chercher si il y a des voisins dans un rayon de 30 px à gauche
            idx_gauche = tree.query_ball_point(p, 30)
            voisins_gauche = [points_list[i] for i in idx_gauche if points_list[i][0] < p[0] - 2]
            #verifier qu'il y a bien une suite à droite (pour s'assurer que c'est une courbe)
            #idx_droite = tree.query_ball_point(p, 30)
            #voisins_droite = [points_list[i] for i in idx_droite if points_list[i][0] > p[0] + 2]
            if len(voisins_gauche) == 0:
                p_suivi = p
                points_suivis = set()
                est_une_chaine = True

                for _ in range(longueur_chaine_min):
                    #cherche le point le plus proche à droite
                    dist, idx = tree.query([p_suivi[0] + 3, p_suivi[1]], k=5)
                    #on filtre pour ne prendre que les points à droite
                    candidats_suivi = [points_list[j] for j in idx if points_list[j][0] > p_suivi[0]]

                    if candidats_suivi:
                        p_suivi = min(candidats_suivi, key=lambda pt: ((pt[0]-p_suivi[0])**2 + (pt[1]-p_suivi[1])**2))
                        #points_suivis.add(p_suivi)
                    else:
                        est_une_chaine = False
                        break #la chaîne casse trop tôt, c'est un débris

                if est_une_chaine:
                    candidats_depart.append(p)

        

    if candidats_depart:
        current_point = min(candidats_depart, key =lambda p: p[0])
        print(f"Début automatique détecté à : {current_point}")

    else :
        # FORCE LE DÉPART À DROITE (même si la chaîne n'est pas parfaite)
        zone_droite = [p for p in points_list if p[0] > debut_recherhce]
        if zone_droite:
            current_point = min(zone_droite, key=lambda p: p[0])
            print(f"Début forcé dans zone 13/09 à : {current_point}")
        else:
            current_point = min(points_list, key=lambda p: p[0])
            print("Début par défaut (bord gauche)")

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
        #calculer la pente sur les 60 derniers points
        if len(x_final) > 60:
            p = np.polyfit(x_final[-60:], y_final[-60:], 1)
            pente = p[0]
        
        #predire prochain point avec inertie
        #cible_x = x_curr + 3
        #cible_y = y_curr + (pente*3)

        #logique j+1
        next_pt = None
        seuil_isolement = 10
        #si proche du bord droit
        if x_curr > largeur_image - 40:
            #on cherche au bord gauche (x=5) à la même hauteur
            dist, idx = tree.query([10, y_curr], k=50) #regarder les 50 plus proches
            candidats = [points_list[i] for i in idx if points_list[i] in points_set]
            if candidats:
                next_pt = min(candidats, key=lambda p: abs(p[1] - y_curr))
                jour_actuel += 1
                print(f"passage au jour {jour_actuel+1}")
            
        if next_pt is None :
            for rayon in [30, 70, 150, 300]:
                #chercher les points existants autour du point souhaité
                dist, idx = tree.query([x_curr, y_curr], k=rayon) #les 100 plus proches
                candidats = [points_list[i] for i in idx if points_list[i] in points_set]

                for avance_min in [2, 1, 0]:
                    candidats_valides = [p for p in candidats if p[0] >= x_curr + avance_min]
                    if candidats_valides:
                        def score_trajectoire(pt_test):
                            #calculer l'écart vertical avec la pente prédite
                    
                            d = ((pt_test[0]-x_curr)**2 + (pt_test[1]-y_curr)**2)**0.5

                            diff_pente = abs(pt_test[1] - (y_curr + pente))
                            #on veut un point proche ET dans la bonne direction
                            #on donne bcp de poids à la direction (x10)
                            return d + (diff_pente * 100)
                
                        next_pt = min(candidats_valides, key=score_trajectoire)
                        break
                if next_pt:
                    break

            #logique de survie (si bloqué au milieu de la feuille)
            # Si on n'a rien trouvé mais qu'on n'est pas encore au bord droit
            if next_pt is None and x_curr < largeur_image - 100:
                print(f"Trou détecté à X={x_curr}. Tentative de saut de secours...")
                # On cherche beaucoup plus loin (400px) à la hauteur estimée
                dist, idx = tree.query([x_curr + 400, y_curr + (pente * 400)], k=100)
                candidats_secours = [points_list[i] for i in idx if points_list[i] in points_set]
                if candidats_secours:
                    # On prend le plus proche de la prédiction de hauteur
                    next_pt = min(candidats_secours, key=lambda p: abs(p[1] - (y_curr + (pente * 400))))
                    print(f"Saut réussi ! Reprise à X={next_pt[0]}")

        if next_pt:
            #if next_pt not in points_set:
                #print("cycle terminé, retour au point de départ du jour 1")
                #break
            current_point = next_pt
            #test pour stopper la lecture de la courbe au bon endroit
            heure_actuelle = (current_point[0] / largeur_image) * 24
            if jour_actuel >=6 and heure_actuelle >= 17.15:
                print("fin du marégramme")
                break
        else:
            print(f"arret à {x_curr} {y_curr}, points restant dans le set : {len(points_set)}")
            print(f"DEBUG : Aucun point trouvé dans un rayon de 300px autour de {x_curr},{y_curr}")
            break

    #x=np.array(x_reconstruit)
    #y=np.array(y_reconstruit)

    #y_smooth = medfilt(y, kernel_size=11)
    
    return np.array(x_final), np.array(y_final), medfilt(y_final, kernel_size=101)

chemin = "image/HPSC0869.tif"

try:
    x_val, y_raw, y_final = extraction_reconstruction_test1(chemin)

    plt.figure(figsize=(12, 6))
    #plt.scatter(x_val, y_raw, s=1, color='gray', alpha=0.5, label='Points bruts (Pixels)')
    plt.plot(x_val, y_final, color='blue', label='Signal lissé (Combo 2)')
    plt.title("Numérisation via Combo 2 (OpenCV + Filtrage + KDTree + suivi de pente)")
    #plt.gca().invert_yaxis() 
    plt.savefig("resultat_numerisation.png")
    print("resultat enregistré")

except Exception as e:
    print(f"Erreur : {e}.")