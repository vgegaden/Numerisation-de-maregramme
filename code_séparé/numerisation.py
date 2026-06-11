import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.spatial import KDTree
import cv2  # Nécessaire pour la fonction cv2.findContours

def reconstruction_maregramme(squelette, img_cropped):
    """
    Prend en entrée le squelette extrait et la grille rognée correspondante.
    Effectue le suivi temporel intelligent courbe par courbe.
    """
    # extraction coord 
    contours, _ = cv2.findContours(squelette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points_list = []
    for i in contours:
        if cv2.arcLength(i, True) > 50:
            for point in i:
                points_list.append(tuple(point[0]))
    
    if not points_list:
        return [0], [0], [0], []
    
    points_set = set(points_list)
    largeur_image = img_cropped.shape[1]

    # creer un index spatial pour la recherche rapide
    tree = KDTree(points_list)

    # point le plus a gauche par défaut
    current_point = min(points_list, key=lambda p: p[0])

    # logique recherche courbe jour 1
    seuil_iso = 100 # distance recherche voisin
    candidats_depart = []
    longueur_chaine_min = 125

    for i in range(0, len(points_list), 10):
        p = points_list[i] 
        idx_gauche = tree.query_ball_point(p, seuil_iso)
        voisins_gauche = [points_list[i] for i in idx_gauche if points_list[i][0] < p[0] - 5]
        
        if len(voisins_gauche) == 0:
            p_suivi = p
            est_une_chaine = True

            for _ in range(longueur_chaine_min):
                dist, idx = tree.query([p_suivi[0] + 10, p_suivi[1]], k=5)
                candidats_suivi = [points_list[j] for j in idx if points_list[j][0] > p_suivi[0]]

                if candidats_suivi:
                    p_suivi = candidats_suivi[0]
                else:
                    est_une_chaine = False
                    break 

            if est_une_chaine:
                candidats_depart.append(p)

    if candidats_depart:
        debut = [p for p in candidats_depart if p[0] > 500]
        if debut:
            current_point = min(debut, key=lambda p: p[0])
        else:
            current_point = min(candidats_depart, key=lambda p: p[0])
        print(f"Début automatique détecté à : {current_point}")
    else:
        current_point = min(points_list, key=lambda p: p[0])
        print("Départ par défaut au bord gauche)")

    jour_actuel = 0
    x_final, y_final = [], []
    iterations = 0
    max_iterations = 50000

    while True:
        x_curr, y_curr = current_point 
        x_final.append(x_curr + (jour_actuel * largeur_image))
        y_final.append(y_curr)

        if current_point in points_set:
            points_set.remove(current_point)

        # logique régression
        pente_lineaire = 0
        poly_courbe = None
        
        if len(x_final) > 30:
            if len(set(x_final[-40:])) > 2:
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
        
        # logique j+1
        next_pt = None
        if x_curr > largeur_image - 100:
            idx = tree.query_ball_point([20, y_curr], 300) 
            candidats = [points_list[i] for i in idx if points_list[i] in points_set]
            if candidats:
                next_pt = min(candidats, key=lambda p: abs(p[1] - y_curr))
                jour_actuel += 1
                x_final = x_final[:-10]
                y_final = y_final[:-10]
                print(f"passage au jour {jour_actuel+1}")
            
        if next_pt is None:
            for rayon in [15, 30, 60]:
                idx = tree.query_ball_point([x_curr, y_curr], rayon) 
                candidats_valides = [points_list[i] for i in idx if points_list[i] in points_set and points_list[i][0] >= x_curr and points_list[i] != current_point]
                
                if candidats_valides:
                    def score_trajectoire(pt_test):
                        dx = max(0.1, abs(pt_test[0] - x_curr))
                        d = ((pt_test[0]-x_curr)**2 + (pt_test[1]-y_curr)**2)**0.5
                        y_predit = y_curr + (pente_lineaire * (pt_test[0] - x_curr))
                        diff_pente_lineaire = abs(pt_test[1] - y_predit) / dx
                        return d + (diff_pente_lineaire * 10)
                
                    next_pt = min(candidats_valides, key=score_trajectoire)
                    break
            
        # logique de survie (si bloqué au milieu de la feuille)
        if next_pt is None and x_curr < largeur_image - 100:
            print(f"Trou détecté à X={x_curr}. Tentative de saut de secours")
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

                if len(set(x_final[-10:])) > 2:
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
            
            for pas_x in range(3, 60, 5):
                cible_y = y_curr + (pente_lineaire * pas_x)
                candidats = chercher_candidats(pas_x, cible_y, 20)
                if candidats:
                    best = min(candidats, key=lambda c: abs(c[0][1] - cible_y))
                    next_pt = best[0]
                    jour_actuel = best[1]
                    print(f"Pont court réussi à x={next_pt[0]}, jour {jour_actuel+1}")
                    break

            if next_pt is None and poly_courbe is not None:
                print("Le pont court a échoué. Tentative de grand saut parabolique.")
                for dist_saut in [60, 150, 400, 600, 1000, 1500, 2000]:
                    cible_y_pure = poly_courbe[0]*(dist_saut**2) + poly_courbe[1]*dist_saut + poly_courbe[2]
                    cible_y = np.clip(cible_y_pure, 200, 3100)
                    rayon_recherche = 80 if dist_saut <= 300 else 200
                    candidats = chercher_candidats(dist_saut, cible_y, rayon_recherche)

                    if poly_courbe is not None:
                        y_par = poly_courbe[0]*(dist_saut**2) + poly_courbe[1]*dist_saut + poly_courbe[2]
                        y_par = np.clip(y_par, 200, 3100)
                        rayon_par = 150 if dist_saut > 300 else 80
                        candidats += chercher_candidats(dist_saut, y_par, rayon_par)
                    
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
                        jour_actuel = best[1]
                        print(f"Grand saut parabolique réussi à x={next_pt[0]}, jour {jour_actuel+1}")
                        break

        if next_pt:
            current_point = next_pt
            heure_actuelle = (current_point[0] / largeur_image) * 24
            if jour_actuel >= 6 and heure_actuelle >= 17.15:
                print("fin du marégramme")
                break
        else:
            print(f"arret à {x_curr} {y_curr}, points restant dans le set : {len(points_set)}")
            break

        iterations += 1
        if iterations >= max_iterations:
            print("nb max d'itérations atteintes")
            break

    return np.array(x_final), np.array(y_final), medfilt(y_final, kernel_size=101), points_list


# ==========================================================
# PARTIE GRAPHIQUES ET PLOTS (À METTRE À LA SUITE DANS LE SCRIPT)
# ==========================================================

# Variables témoins nécessaires pour dimensionner les graphiques journaliers
# (Doivent être récupérées depuis ton image rognée "img" ou "test_grille_isolee.png")
largeur_image = img.shape[1]
hauteur_image = img.shape[0]
points_list_np = np.array(points_list)

# --- 1. PLOT GLOBAL (Signal complet sur toute la période) ---
plt.figure(figsize=(12, 6))
plt.plot(x_val, y_final, color='blue', label='Signal lissé (Combo 2)')
plt.scatter(x_val % 4722, y_raw, s=1, color='red', alpha=0.1, label='Points bruts')
plt.title("Numérisation avec OpenCV + Filtrage + KDTree + suivi de pente_lineaire)")
plt.gca().invert_yaxis() 
plt.legend()
plt.savefig("resultat_numerisation.png")
plt.close()
print("resultat global enregistré")

# Tronquage des derniers points de sécurité pour éviter les anomalies de bordure sur les graphs
x_val = x_val[:-10]
y_raw = y_raw[:-10]
y_final = y_final[:-10]

# Détermination automatique du nombre de jours détectés
jours_associes = (x_val // largeur_image).astype(int)
nb_jours_trouves = jours_associes.max() + 1

# --- 2. PLOTS INDIVIDUELS (Jour par Jour de 00h à 24h) ---
for j in range(nb_jours_trouves):
    masque_jour = (jours_associes == j)
    if not np.any(masque_jour):
        continue

    x_jour_local = x_val[masque_jour] % largeur_image
    y_jour_brut = y_raw[masque_jour]

    plt.figure(figsize=(12, 6))
    diffs = np.diff(x_jour_local)
    y_visualisation = y_jour_brut.astype(float).copy()
    
    # Nettoyage visuel : évite de tracer des lignes géantes horizontales lors des sauts temporels
    indices_sauts = np.where((diffs < 0) | (diffs > 100))[0]
    for idx in indices_sauts:
        y_visualisation[idx] = np.nan
    
    # Tracé de la courbe reconstruite du jour
    plt.plot(x_jour_local, y_visualisation, color='blue', linewidth=2, linestyle='-', label=f'Reconstruction Jour {j+1}', zorder=5)
    
    # Superposition en tâche de fond gris clair de tous les points du squelette extrait par OpenCV
    plt.scatter(points_list_np[:, 0], points_list_np[:, 1], s=1, color='lightgray', alpha=0.1, label='Squelette total')

    # Configuration de l'axe des X pour afficher les heures réelles (pas de 2 heures)
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
    
    # Enregistrement dynamique de l'image (ex: graph_jour1.png, graph_jour2.png...)
    nom_fichier = f"graph_jour{j+1}.png"
    plt.savefig(nom_fichier)
    plt.close()
    
    print(f"Graphique {nom_fichier} enregistré.")