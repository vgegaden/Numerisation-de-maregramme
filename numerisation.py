#interaction utilisateur pour demander la date
#extraction grille avec detection des croisements des lignes de fond
#extraction courbes avec hsv
#recherche du point de départ avec un kdtree
#gestion des croisements avec régression linéaire
#retour à la ligne pour gérer la courbe du jour suivant


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.spatial import KDTree #pour améliorer la vitesse
import traceback
import os
import random
from datetime import datetime


def etape_0_acquisition_dates():
    
    # 1. Demande de la date de pose
    while True:
        saisie_pose = input("Entrez la date de POSE (format DD/MM/YYYY) : ")
        try:
            date_pose = datetime.strptime(saisie_pose, "%d/%m/%Y")
            break
        except ValueError:
            print("Format incorrect. Utiliser le format JJ/MM/AAAA (ex: 27/01/1909) et sans espace.")

    # 2. Demande de la date de retrait
    while True:
        saisie_retrait = input("Entrez la date de RETRAIT (format DD/MM/YYYY) : ")
        try:
            date_retrait = datetime.strptime(saisie_retrait, "%d/%m/%Y")
            if date_retrait < date_pose:
                print("Erreur : Date de retrait antérieure à la date de pose.")
                continue
            break
        except ValueError:
            print("Format incorrect. Utiliser le format JJ/MM/AAAA (ex: 27/01/1909) et sans espace.")

    # 3. Calcul de l'écart
    # .days renvoie la différence exacte en nombre de jours
    nb_jours = (date_retrait - date_pose).days + 1
    
    # ATTENTION : Si posé le 27 et retiré le 28, l'écart est de 1 jour, 
    # mais il y a techniquement 2 courbes entamées sur le papier (le 27 ET le 28).
    nb_courbes = nb_jours

    
    print(f"Marégramme allant du {date_pose.strftime('%d/%m/%Y')} au {date_retrait.strftime('%d/%m/%Y')}")
    print(f"Enregistrement de {nb_jours} jours")
    print(f"Nombre de courbes théoriques : {nb_courbes}")
    
    return date_pose, date_retrait, nb_courbes


def redresser_et_rogner_grille(img_hd):
    h, w = img_hd.shape[:2]
    
    # 1. Extraction Structurelle
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
    #
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

    #partie segmentation couleurs
    #bleu_clair = np.array([0, 0, 0])
    #bleu_fonce = np.array([179, 200, 150])
    #masque_bleu = cv2.inRange(hsv, bleu_clair, bleu_fonce)

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

    # 3. On remplace l'ancien squelette
    squelette = squelette_nettoye
    # ------------------------------------------------

    cv2.imwrite("image_apres_extraction/debug_3_squelette.png", squelette)
    return squelette


def reconstruction_maregramme(chemin_img):
    img_hd = cv2.imread(chemin_img)
    img = redresser_et_rogner_grille(img_hd)
    cv2.imwrite("test_grille_isolee.png", img)
    
    # Appel de la fonction d'extraction
    squelette = extraction_courbe_hsv(img)

    #extraction coord 
    contours, _ = cv2.findContours(squelette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points_list = []
    for i in contours:
        if cv2.arcLength(i, True) > 50:
            for point in i:
                points_list.append(tuple(point[0]))
    
    if not points_list:
        return [0],[0],[0], [], []
    
    points_set = set(points_list)
    x_reconstruit = []
    y_reconstruit = []

    #creer un index spatial pour la recherche rapide
    tree = KDTree(points_list)


    #point le plus a gauche
    current_point = min(points_list, key =lambda p: p[0])
    largeur_image = img.shape[1]

    #logique recherche courbe jour 1
    seuil_iso = 100 #distance recherche voisin
    candidats_depart = []
    longueur_chaine_min = 125

    #debut_recherhce = int(largeur_image*0.6)

    for i in range (0, len(points_list), 10) :
        p = points_list[i] 
        #if p[0] > debut_recherhce: #50px pour ne pas commencer au bord
            #chercher si il y a des voisins dans un rayon de 30 px à gauche
        idx_gauche = tree.query_ball_point(p, seuil_iso)
        voisins_gauche = [points_list[i] for i in idx_gauche if points_list[i][0] < p[0] - 5]
            #verifier qu'il y a bien une suite à droite (pour s'assurer que c'est une courbe)
            #idx_droite = tree.query_ball_point(p, 30)
            #voisins_droite = [points_list[i] for i in idx_droite if points_list[i][0] > p[0] + 2]
        if len(voisins_gauche) == 0:
            p_suivi = p
            points_suivis = set()
            est_une_chaine = True

            for _ in range(longueur_chaine_min):
                #cherche le point le plus proche à droite
                dist, idx = tree.query([p_suivi[0] + 10, p_suivi[1]], k=5)
                #on filtre pour ne prendre que les points à droite
                candidats_suivi = [points_list[j] for j in idx if points_list[j][0] > p_suivi[0]]

                if candidats_suivi:
                    p_suivi = candidats_suivi[0]
                    #points_suivis.add(p_suivi)
                else:
                    est_une_chaine = False
                    break #la chaîne casse trop tôt, c'est un débris

            if est_une_chaine:
                candidats_depart.append(p)

        

    if candidats_depart:
        #eliminer les candidats trop proches du bord pour logique courbe jour 1 (à remodifier plus tard)
        debut = [p for p in candidats_depart if p[0] > 500]
        if debut:
            current_point = min(debut, key =lambda p: p[0])
        else : 
            current_point = min(candidats_depart, key = lambda p : p[0])
        print(f"Début automatique détecté à : {current_point}")

    else :
        #depart par defaut à gauche
        current_point = min(points_list, key=lambda p : p[0])
        print("Départ par défaut au bord gauche)")

    #centre_image = largeur_image // 2 
    #pixels du bord
    #seuil_bord = 15
    #nb de points pour la pente_lineaire
    #nb_points_reg = 15
    #cpt de sauts j+1
    jour_actuel = 0
    x_final, y_final = [], []
    #test
    points_restants = set(points_list)
    iterations = 0
    max_iterations = 50000
    visualisation_pentes = []

    while True:
        x_curr, y_curr = current_point 
        x_final.append(x_curr + (jour_actuel * largeur_image))
        y_final.append(y_curr)

        
        if current_point in points_set:
            points_set.remove(current_point)

        #logique régression
        pente_lineaire = 0
        poly_courbe = None
        

        #nb = min(len(x_final), 40)
        if len(x_final) > 30:
            if len(set(x_final[-40:])) >2:
                x_recent_lin = np.array(x_final[-40:])
                y_recent_lin = np.array(y_final[-40:])
                x_ref_lin = x_recent_lin[-1]
                p_lin = np.polyfit(x_recent_lin - x_ref_lin, y_recent_lin, 1)

                pente_lineaire = p_lin[0]
            if len(set(x_final[-200:])) > 10:
                x_recent = np.array(x_final[-200:])
                y_recent = np.array(y_final[-200:]) 
                x_ref = x_recent[-1]   
                poly_courbe = np.polyfit(x_recent - x_ref, y_recent, 2)
        
        #predire prochain point avec inertie
        #cible_x = x_curr + 3
        #cible_y = y_curr + (pente_lineaire*3)

        #logique passage au jour suivant
        next_pt = None
        seuil_isolement = 10
        #si proche du bord droit
        if x_curr > largeur_image - 100:
            #on cherche au bord gauche (x=5) à la même hauteur
            idx = tree.query_ball_point([20, y_curr], 300) #regarder les 50 plus proches
            candidats = [points_list[i] for i in idx if points_list[i] in points_set]
            if candidats:
                next_pt = min(candidats, key=lambda p: abs(p[1] - y_curr))
                jour_actuel += 1
                x_final = x_final[:-10]
                y_final = y_final[:-10]
                print(f"passage au jour {jour_actuel+1}")
            
        if next_pt is None :
            for rayon in [15, 30, 60]:
                #chercher les points existants autour du point souhaité
                idx = tree.query_ball_point([x_curr, y_curr], rayon) 

                #for avance_min in [2, 1, 0]:
                #if candidats:

                candidats_valides = [points_list[i] for i in idx if points_list[i] in points_set and points_list[i][0] >= x_curr and points_list[i] != current_point]
                
                if candidats_valides:

                    if len(candidats_valides) > 1:
                        x_abs_start = x_curr + (jour_actuel * largeur_image)
                        y_start = y_curr
                        dx = 60
                        x_abs_end = x_abs_start + dx
                        y_end = y_curr + (pente_lineaire * dx)

                        visualisation_pentes.append(([x_abs_start, x_abs_end], [y_start, y_end]))
                    def score_trajectoire(pt_test):
                        #calculer l'écart vertical avec la pente_lineaire prédite
                        dx = max(0.1, abs(pt_test[0] - x_curr))
                        d = ((pt_test[0]-x_curr)**2 + (pt_test[1]-y_curr)**2)**0.5
                        
                        y_predit = y_curr + (pente_lineaire * (pt_test[0] - x_curr))
                        diff_pente_lineaire = abs(pt_test[1] - y_predit) / dx
                        #on veut un point proche ET dans la bonne direction
                        #on donne bcp de poids à la direction (x10)
                        return d + (diff_pente_lineaire * 10)
                
                    next_pt = min(candidats_valides, key=score_trajectoire)
                    break
            

            
            
            
        #logique de survie (si bloqué au milieu de la feuille)
        # Si on n'a rien trouvé mais qu'on n'est pas encore au bord droit
        if next_pt is None and x_curr < largeur_image - 100:
            print(f"Trou détecté à X={x_curr}. Tentative de saut de secours")
            # On cherche beaucoup plus loin (400px) à la hauteur estimée
            
            nb_recul = 15
            if len(x_final) > nb_recul + 40:
                for _ in range(nb_recul):
                    if len(x_final) >= 2:
                        dist_x = abs(x_final[-1] - x_final[-2])
                        
                        if dist_x > 80:
                            x_final.pop()
                            y_final.pop()
                            print("saut annulé, sauvegarde de la piste de décollage")
                            break

                        x_final.pop()
                        y_final.pop()
                    else:
                        break
                jour_actuel = int(x_final[-1] // largeur_image)
                x_curr = x_final[-1] - (jour_actuel * largeur_image)
                y_curr = y_final[-1]

                if len(set(x_final[-10:])) >2:
                    p_lin = np.polyfit(x_final[-10:], y_final[-10:], 1)
                    pente_lineaire = p_lin[0]
                if len(set(x_final[-40:])) > 3:
                    x_recent = np.array(x_final[-40:])
                    y_recent = np.array(y_final[-40:])
                    x_ref = x_recent[-1]
                    poly_courbe = np.polyfit(x_recent - x_ref, y_recent, 2)
                
                
                
                print(f"Recul effectué. Reprise à X={x_curr}")

                def chercher_candidats(dist_saut, y_cible, rayon):
                    x_abs_cible = x_final[-1] + dist_saut
                    x_loc_cible = x_abs_cible % largeur_image
                    j_cible = int(x_abs_cible // largeur_image)
                
                    idx = tree.query_ball_point([x_loc_cible, y_cible], rayon)
                    valides = []
                    for i in idx:
                        p = points_list[i]
                        if p in points_set:
                            p_abs = p[0] + (j_cible * largeur_image)
                            if p_abs > x_final[-1] + 10:
                                valides.append((p, j_cible, p_abs))
                    return valides
            
            for pas_x in range (3, 60, 5):

                #cible_x = x_curr + pas_x
                cible_y = y_curr + (pente_lineaire * pas_x)
                #idx = tree.query_ball_point([cible_x, cible_y], 15)
                #candidats_secours = [points_list[i] for i in idx if points_list[i] in points_set and points_list[i][0] > x_curr]
                candidats = chercher_candidats(pas_x, cible_y, 20)
                if candidats:
                    # On prend le plus proche de la prédiction de hauteur
                    best = min(candidats, key=lambda c: abs(c[0][1] - cible_y))
                    next_pt = best[0]
                    jour_actuel = best[1]
                    print(f"Pont court réussi à x={next_pt[0]}, jour {jour_actuel+1}")
                    break

            if next_pt is None and poly_courbe is not None:
                print("Le pont court a échoué. Tentative de grand saut parabolique.")
                for dist_saut in [60, 150, 400, 600, 1000, 1500, 2000]:
                    #cible_x = x_curr + dist_saut
                    #cible_x_absolu = x_final[-1] + dist_saut
                    cible_y_pure = poly_courbe[0]*(dist_saut**2) + poly_courbe[1]*dist_saut + poly_courbe[2]
                    cible_y = np.clip(cible_y_pure, 200, 3100)
                    rayon_recherche = 80 if dist_saut <= 300 else 200
                    #idx = tree.query_ball_point([cible_x, cible_y], 100) 
                    candidats = chercher_candidats(dist_saut, cible_y, rayon_recherche)

                    y_par = None
                    if poly_courbe is not None:
                        y_par = poly_courbe[0]*(dist_saut**2) + poly_courbe[1]*dist_saut + poly_courbe[2]
                        y_par = np.clip(y_par, 200, 3100)
                        rayon_par = 150 if dist_saut > 300 else 80
                        # On ajoute les candidats de la parabole à notre liste de recherche
                        candidats += chercher_candidats(dist_saut, y_par, rayon_par)
                    
                    # Dédoublonnage (si la ligne et la parabole pointent au même endroit)
                    candidats_uniques = list({c[0]: c for c in candidats}.values())
                    
                    candidats_solides = []
                    for c in candidats:
                        p_loc = c[0]
                        voisins = tree.query_ball_point(p_loc, 10)
                        voisins_vivants = [v for v in voisins if points_list[v] in points_set]
                        if len(voisins_vivants) >= 5:
                            candidats_solides.append(c)

                    if candidats_solides:
                        def score_atterissage(c):
                            return abs(c[0][1] - cible_y)

                        best = min(candidats_solides, key=score_atterissage)
                        next_pt = best[0]
                        jour_actuel=best[1]
                        print(f"Grand saut parabolique réussi à x={next_pt[0]}, jour {jour_actuel+1}")
                        break
                

        #points_visites_total = set()
        if next_pt:
            #if next_pt not in points_set:
                #print("cycle terminé, retour au point de départ du jour 1")
                #break
            current_point = next_pt
            #test pour stopper la lecture de la courbe au bon endroit (a supprimer)
            heure_actuelle = (current_point[0] / largeur_image) * 24
            if jour_actuel >=6 and heure_actuelle >= 17.15:
                print("fin du marégramme")
                break
            

        else:
            print(f"arret à {x_curr} {y_curr}, points restant dans le set : {len(points_set)}")
            print(f"DEBUG : Aucun point trouvé dans un rayon de 300px autour de {x_curr},{y_curr}")
            break

        iterations += 1

        if iterations >= max_iterations:
            print("nb max d'itérations atteintes")

    #x=np.array(x_reconstruit)
    #y=np.array(y_reconstruit)

    #y_smooth = medfilt(y, kernel_size=11)
    
    return np.array(x_final), np.array(y_final), medfilt(y_final, kernel_size=101), visualisation_pentes, points_list


# ==========================================
# SCRIPT PRINCIPAL D'EXECUTION ET GRAPHES
# ==========================================
if __name__ == "__main__":
    
    # Étape 0 : Acquisition des dates de pose et retrait en console
    d_pose, d_retrait, courbes_theoriques = etape_0_acquisition_dates()
    
    chemin = "image/HPSC0178.tif"

    try:
        # Appel de la fonction renommée principale
        x_val, y_raw, y_final, v_pentes, points_list = reconstruction_maregramme(chemin)
        points_list_np = np.array(points_list)


        plt.figure(figsize=(12, 6))
        #plt.scatter(x_val, y_raw, s=1, color='gray', alpha=0.5, label='Points bruts (Pixels)')
        plt.plot(x_val, y_final, color='blue', label='Signal lissé (Combo 2)')

        plt.scatter(x_val % 4722, y_raw, s=1, color='red', alpha=0.1, label='Points bruts')

        plt.title("Numérisation avec OpenCV + Filtrage + KDTree + suivi de pente_lineaire)")
        plt.gca().invert_yaxis() 
        plt.legend()
        plt.savefig("resultat_numerisation.png")
        plt.close()
        print("resultat enregistré")

        x_val = x_val[:-10]
        y_raw = y_raw[:-10]
        y_final = y_final[:-10]
        
        img = cv2.imread(chemin)
        largeur_image = img.shape[1]
        hauteur_image = img.shape[0]
        jours_associes = (x_val // largeur_image).astype(int)
        nb_jours_trouves = jours_associes.max() + 1

        for j in range(nb_jours_trouves):
            masque_jour = (jours_associes == j)
            if not np.any(masque_jour):
                continue

            x_jour_local = x_val[masque_jour] % largeur_image
            y_jour_brut = y_raw[masque_jour]

            plt.figure(figsize=(12, 6))
            diffs = np.diff(x_jour_local)
            y_visualisation = y_jour_brut.astype(float).copy()
            indices_sauts = np.where((diffs < 0) | (diffs > 100))[0]
            for idx in indices_sauts:
                y_visualisation[idx] = np.nan
            
            plt.plot(x_jour_local, y_visualisation, color='blue', linewidth=2, linestyle='-', label=f'Reconstruction Jour {j+1}', zorder=5)
            plt.scatter(points_list_np[:, 0], points_list_np[:, 1], s=1, color='lightgray', alpha=0.1, label='Squelette total')

            first_pente = True
            for (seg_x_abs, seg_y) in v_pentes:
                # On utilise le premier point du segment pour déterminer le jour
                x_debut_abs = seg_x_abs[0]
                jour_du_segment = int(x_debut_abs // largeur_image)
                
                if jour_du_segment == j:
                    # Conversion des X absolus en locaux pour le graphique (0 à largeur_image)
                    x_local = [x % largeur_image for x in seg_x_abs]
                    
                    if abs(x_local[1] - x_local[0]) < (largeur_image / 2):

                        lbl = "Pentes de décision" if first_pente else ""
                        plt.plot(x_local, seg_y, color='red', linewidth=1.5, alpha=0.8, zorder=10, label=lbl)
                    
                        # Petit point vert au départ de la décision
                        plt.scatter(x_local[0], seg_y[0], color='green', s=10, zorder=11)
                        first_pente = False

            #pour afficher les heures sur le graph
            heures_labels = [f"{h:02d}h" for h in range(0, 25, 2)]
            positions_pixels = [(h * largeur_image) / 24 for h in range(0, 25, 2)]
            plt.xticks(positions_pixels, heures_labels)
            plt.xlim(0, largeur_image)
            plt.ylim(0, hauteur_image)
            plt.grid(axis='x', linestyle='--', alpha=0.4)
            
            plt.title(f"Marégramme - Jour {j+1}")
            plt.xlabel("heure")
            plt.ylabel("Pixels (Hauteur)")
            plt.gca().invert_yaxis()
            plt.legend()
            
            # Sauvegarde avec le nom dynamique
            nom_fichier = f"graph_jour{j+1}.png"
            plt.savefig(nom_fichier)
            plt.close()
            
            print(f"Graphique {nom_fichier} enregistré.")
        
    except Exception as e:
        print(f"Erreur : {e}.")
        traceback.print_exc()
