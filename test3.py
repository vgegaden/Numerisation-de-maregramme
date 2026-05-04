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

def extraction_reconstruction_test1(chemin_img):
    img = cv2.imread(chemin_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    #partie segmentation couleurs
    bleu_clair = np.array([0, 0, 0])
    bleu_fonce = np.array([179, 255, 100])
    masque_bleu = cv2.inRange(hsv, bleu_clair, bleu_fonce)

    cv2.imwrite("image_apres_extraction/debug_1_extraction_brute.png", masque_bleu)

    kernel5 = np.ones((5, 5), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel7 = np.ones((7, 7), np.uint8)

    masque_gras= cv2.dilate(masque_bleu, kernel5, iterations = 1)
    masque_propre = cv2.morphologyEx(masque_gras, cv2.MORPH_CLOSE, kernel5)
    #masque_propre = cv2.morphologyEx(masque_propre, cv2.MORPH_OPEN, kernel3)
    masque_propre = cv2.medianBlur(masque_propre, 5)

    #kernel_bouche_trou = np.ones((5, 5), np.uint8)
    #masque_lisse = cv2.morphologyEx(masque_propre, cv2.MORPH_CLOSE, kernel_bouche_trou)


    cv2.imwrite("image_apres_extraction/debug_2_extraction_propre.png", masque_propre)


    #squelettisation
    squelette = cv2.ximgproc.thinning(masque_propre)
    cv2.imwrite("image_apres_extraction/debug_3_squelette.png", squelette)
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
                p_lin = np.polyfit(x_final[-40:], y_final[-40:], 1)
                pente_lineaire = p_lin[0]
            if len(set(x_final[-200:])) > 10:
                x_recent = np.array(x_final[-200:])
                y_recent = np.array(y_final[-200:]) 
                x_ref = x_recent[-1]   
                poly_courbe = np.polyfit(x_recent - x_ref, y_recent, 2)
        
        #predire prochain point avec inertie
        #cible_x = x_curr + 3
        #cible_y = y_curr + (pente_lineaire*3)

        #logique j+1
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
                        x_start = x_curr
                        y_start = y_curr
                        x_end = x_curr + 60
                        y_end = y_curr + (pente_lineaire * 60)

                        visualisation_pentes.append(([x_start, x_end], [y_start, y_end]))
                    def score_trajectoire(pt_test):
                        #calculer l'écart vertical avec la pente_lineaire prédite
                
                        d = ((pt_test[0]-x_curr)**2 + (pt_test[1]-y_curr)**2)**0.5
                        
                        y_predit = y_curr + (pente_lineaire * (pt_test[0] - x_curr))
                        diff_pente_lineaire = abs(pt_test[1] - y_predit)
                        #on veut un point proche ET dans la bonne direction
                        #on donne bcp de poids à la direction (x10)
                        return d + (diff_pente_lineaire * 50)
                
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
            #test pour stopper la lecture de la courbe au bon endroit
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

chemin = "image/HPSC0869.tif"

try:
    x_val, y_raw, y_final, v_pentes, points_list = extraction_reconstruction_test1(chemin)
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

    img = cv2.imread(chemin)
    largeur_image = img.shape[1]
    jours_associes = (x_val // largeur_image).astype(int)
    nb_jours_trouves = jours_associes.max() + 1
    
    for j in range(nb_jours_trouves):
        masque_jour = (jours_associes == j)
        if not np.any(masque_jour):
            continue

        x_jour_local = x_val[masque_jour] % largeur_image
        y_jour_brut = y_raw[masque_jour]

        plt.figure(figsize=(12, 6))
        plt.scatter(x_val % largeur_image, y_raw, s=1, color='gray', alpha=0.05, label='Faisceau brut')
        plt.plot(x_jour_local, y_jour_brut, color='blue', linewidth=2, label=f'Reconstruction Jour {j+1}')
        
        plt.title(f"Marégramme - Jour {j+1}")
        plt.xlabel("Pixels (Temps)")
        plt.ylabel("Pixels (Hauteur)")
        plt.gca().invert_yaxis()
        plt.legend()
        
        # Sauvegarde avec le nom dynamique
        nom_fichier = f"graph_jour{j+1}.png"
        plt.savefig(nom_fichier)
        plt.close()
        
        print(f"Graphique {nom_fichier} enregistré.")
    '''
    plt.figure(figsize=(20, 10))
    
    # --- ÉTAPE 1 : Le fond (Squelette gris) ---
    # On affiche uniquement les points qui appartiennent au Jour 1 (0 < x < largeur_image)
    masque_fond_j1 = points_list_np[:, 0] < largeur_image
    plt.scatter(points_list_np[masque_fond_j1, 0], points_list_np[masque_fond_j1, 1], 
            s=1, color='lightgray', alpha=0.4, label='Squelette Jour 1', zorder=1)

    # --- ÉTAPE 2 : Les segments de pente (Rouge) ---
    # On filtre la liste v_pentes pour ne garder que le Jour 1
    v_pentes_j1 = [p for p in v_pentes if p[0][0] < largeur_image]

    for i, (seg_x, seg_y) in enumerate(v_pentes_j1):
        label = "Pente de décision" if i == 0 else ""
        # On trace le segment rouge très finement
        plt.plot(seg_x, seg_y, color='red', linewidth=1.2, alpha=0.9, zorder=3, label=label)
    
        # On ajoute un petit point vert au départ du segment (l'endroit du choix)
        plt.scatter(seg_x[0], seg_y[0], color='green', s=5, zorder=4)

    # --- ÉTAPE 3 : Habillage ---
    plt.title("Analyse des Croisements - Jour 1 : Squelette vs Pente Prédictive")
    plt.xlabel("Pixels X")
    plt.ylabel("Pixels Y (Profondeur)")
    plt.gca().invert_yaxis() # Inversion pour avoir le haut en haut
    plt.legend()

    # Sauvegarde en haute qualité pour pouvoir zoomer sur les croisements
    plt.savefig("diagnostic_croisements_J1.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graphique enregistré : diagnostic_croisements_J1.png ({len(v_pentes_j1)} intersections analysées)")
    '''
except Exception as e:
    print(f"Erreur : {e}.")
    traceback.print_exc()

