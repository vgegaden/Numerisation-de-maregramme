import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.spatial import KDTree
import traceback

def extraction_reconstruction_test1(chemin_img):
    img = cv2.imread(chemin_img)
    if img is None: return [0],[0],[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 1. Extraction et Squelettisation ---
    bleu_clair = np.array([0, 0, 0])
    bleu_fonce = np.array([179, 255, 100])
    masque_brut = cv2.inRange(hsv, bleu_clair, bleu_fonce)

    # Nettoyage : suppression des petits objets (chiffres, taches < 100 pixels)
    nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masque_brut)
    masque_propre = np.zeros_like(masque_brut)
    for i in range(1, nb_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100:
            masque_propre[labels == i] = 255

    kernel7 = np.ones((7, 7), np.uint8)
    masque_gras = cv2.dilate(masque_propre, kernel7, iterations=1)
    masque_final = cv2.morphologyEx(masque_gras, cv2.MORPH_CLOSE, kernel7)

    squelette = cv2.ximgproc.thinning(masque_final)
    contours, _ = cv2.findContours(squelette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    points_list = []
    for i in contours:
        if cv2.arcLength(i, True) > 50:
            for point in i:
                points_list.append(tuple(point[0]))
    
    if not points_list: return [0],[0],[0]
    
    points_set = set(points_list)
    tree = KDTree(points_list)
    largeur_image = img.shape[1]

    # --- 2. DÉPART DYNAMIQUE ---
    seuil_iso = 100
    candidats_depart = []
    for i in range (0, len(points_list), 10):
        p = points_list[i]
        voisins_gauche = [points_list[j] for j in tree.query_ball_point(p, seuil_iso) if points_list[j][0] < p[0] - 5]
        if len(voisins_gauche) == 0:
            p_suivi = p
            est_une_chaine = True
            for _ in range(12):
                dist, idx = tree.query([p_suivi[0] + 10, p_suivi[1]], k=5)
                suivants = [points_list[j] for j in idx if points_list[j][0] > p_suivi[0]]
                if suivants: p_suivi = suivants[0]
                else:
                    est_une_chaine = False
                    break
            if est_une_chaine: candidats_depart.append(p)

    if candidats_depart:
        debut = [c for c in candidats_depart if c[0] > 500]
        current_point = min(debut, key=lambda c: c[0]) if debut else min(candidats_depart, key=lambda c: c[0])
    else:
        current_point = min(points_list, key=lambda p: p[0])

    # --- 3. BOUCLE DE PRÉDICTION AVEC GESTION DES CROISEMENTS ---
    jour_actuel = 0
    x_final, y_final = [], []

    while True:
        x_curr, y_curr = current_point
        x_final.append(x_curr + (jour_actuel * largeur_image))
        y_final.append(y_curr)

        if current_point in points_set:
            points_set.remove(current_point)

        next_pt = None
        jour_cible = jour_actuel

        assez_de_recul_X = False
        if len(x_final) >= 20:
            recul_horizontal = x_final[-1] - x_final[-20]
            if recul_horizontal >= 15:
                assez_de_recul_X = True

        if assez_de_recul_X:
            nb_pts = min(len(x_final), 150)
            x_recent = np.array(x_final[-nb_pts:]) - x_final[-1] 
            y_recent = np.array(y_final[-nb_pts:])
            poly = np.polyfit(x_recent, y_recent, 2)

            # Vecteur de direction actuel (Inertie)
            v_dir = np.array([x_final[-1] - x_final[-15], y_final[-1] - y_final[-15]])
            norm_v = np.linalg.norm(v_dir)

            for saut in [15, 30, 60, 150, 300, 600, 1000]:
                y_cible = poly[0]*(saut**2) + poly[1]*saut + poly[2]
                x_abs_cible = x_final[-1] + saut
                x_loc_cible = x_abs_cible % largeur_image
                j_cible_pot = int(x_abs_cible // largeur_image)
                
                rayon_recherche = 15 if saut <= 30 else (40 if saut <= 150 else 100)
                idx = tree.query_ball_point([x_loc_cible, y_cible], rayon_recherche)
                
                scores_candidats = []
                for i in idx:
                    p = points_list[i]
                    if p in points_set:
                        p_abs = p[0] + (j_cible_pot * largeur_image)
                        if p_abs > x_final[-1] + (saut * 0.5):
                            # Calcul de la déviation angulaire
                            v_cand = np.array([p_abs - x_final[-1], p[1] - y_final[-1]])
                            norm_c = np.linalg.norm(v_cand)
                            penalite_angle = 1.0
                            if norm_c > 0 and norm_v > 0:
                                cos_theta = np.dot(v_dir, v_cand) / (norm_v * norm_c)
                                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                                if angle > 0.8: # Environ 45 degrés
                                    penalite_angle = 50.0 # Rejet quasi-systématique
                                else:
                                    penalite_angle = 1.0 + angle
                            
                            dist_pred = abs(p[1] - y_cible)
                            score = dist_pred * penalite_angle
                            scores_candidats.append((score, p, j_cible_pot))
                
                if scores_candidats:
                    best = min(scores_candidats, key=lambda x: x[0])
                    next_pt, jour_cible = best[1], best[2]
                    break
        else:
            # Phase de démarrage
            for rayon in [15, 30, 60]:
                idx = tree.query_ball_point([x_curr, y_curr], rayon)
                candidats = [points_list[i] for i in idx if points_list[i] in points_set and points_list[i][0] >= x_curr - 2 and points_list[i] != current_point]
                if candidats:
                    next_pt = min(candidats, key=lambda p: ((p[0]-x_curr)**2 + (p[1]-y_curr)**2)**0.5)
                    break

        if next_pt:
            current_point, jour_actuel = next_pt, jour_cible
            heure_actuelle = (current_point[0] / largeur_image) * 24
            if jour_actuel >= 6 and heure_actuelle >= 17.15: break
        else: break

    taille_filtre = 11 if len(y_final) > 11 else (3 if len(y_final) > 3 else 1)
    return np.array(x_final), np.array(y_final), medfilt(y_final, kernel_size=taille_filtre)

# ==========================================
# Exécution et Graphiques
# ==========================================
chemin = "image/HPSC0869.tif"

try:
    x_val, y_raw, y_final = extraction_reconstruction_test1(chemin)
    largeur_image = 4722

    # Global
    plt.figure(figsize=(12, 6))
    plt.plot(x_val, y_final, color='blue', label='Signal continu complet')
    plt.scatter(x_val % largeur_image, y_raw, s=1, color='red', alpha=0.1, label='Points bruts')
    plt.title("Marégramme Complet (Anticipation Parabolique)")
    plt.gca().invert_yaxis() 
    plt.legend()
    plt.savefig("resultat_numerisation.png")
    plt.close()
    print("Graphique global enregistré.")

    # Par jour
    jours_associes = (x_val // largeur_image).astype(int)
    for j in range(jours_associes.max() + 1):
        masque_jour = (jours_associes == j)
        if not np.any(masque_jour): continue

        plt.figure(figsize=(12, 6))
        plt.scatter(x_val % largeur_image, y_raw, s=1, color='gray', alpha=0.05)
        plt.plot(x_val[masque_jour] % largeur_image, y_raw[masque_jour], color='blue', linewidth=2)
        plt.title(f"Marégramme - Jour {j+1}")
        plt.gca().invert_yaxis()
        plt.savefig(f"graph_jour{j+1}.png")
        plt.close()
        print(f"Graphique jour {j+1} enregistré.")

except Exception as e:
    print(f"Erreur : {e}")
    traceback.print_exc()