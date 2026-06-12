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

etape_0_acquisition_dates()