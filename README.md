# Numérisation Automatique de Marégrammes Papier

Ce projet a été réalisé par **Valentin GEGADEN** dans le cadre d'un stage de fin de Licence (L3) à **La Rochelle Université**, au sein des laboratoires **MIA** (Mathématiques, Image et Applications) et **LIENSs** (Littoral, ENvironnement et Sociétés), en collaboration avec les données historiques du **Shom** (Service hydrographique et océanographique de la Marine).

L'objectif principal est de développer un outil/algorithme capable d'automatiser la numérisation des marégrammes papier scannés (fichiers `.tif`) issus du port de Douala (Cameroun), afin de pallier les limites du logiciel semi-automatique actuel, NUNIEAU.

---

## 🚀 Fonctionnalités du Projet

L'algorithme découpe le processus de traitement d'image en plusieurs étapes clés :
* **Identification de la date** : Une interaction console avec l'utilisateur permet de définir les dates de pose et de retrait afin de calculer le nombre théorique de courbes à extraire.
* **Extraction de la grille** : Détection géométrique des lignes horizontales et verticales via leurs intersections pour recadrer, rogner et redresser (transformation affine) le marégramme sans altérer la résolution d'origine.
* **Extraction des courbes** : Segmentation avancée par couleur en utilisant l'espace HSV (Teinte, Saturation, Luminosité) et l'analyse d'histogramme pour isoler automatiquement la teinte dominante de la courbe (bleu foncé/noir) du fond blanc et de la grille.
* **Trouver le point de départ** : Identification du point de départ de la première courbe en s'imaginant que le début d'une courbe est forcément un point qui n'a pas de voisins à sa gauche
* **Suivi des courbes & Gestion des croisements** : Détection des croisements et prédiction de la tendance physique à suivre grâce à un calcul de régression linéaire basé sur l'inertie naturelle de la courbe.
* **Suivi du jour suivant** : Transition automatique d'une courbe vers la colonne X=0 du jour suivant par continuité physique de la marée, en exploitant les arbres à K dimensions (KD-Trees) pour la recherche des voisins les plus proches.

---

## 🛠️ Installation et Configuration

Suivez ces étapes pour installer le projet et configurer votre environnement de développement.

### Prérequis
Assurez-vous d'avoir **Python 3.9+** installé sur votre machine.

### 1. Cloner le projet
```bash
git clone [https://github.com/vgegaden/Numerisation-de-maregramme.git](https://github.com/vgegaden/Numerisation-de-maregramme.git)
cd Numerisation-de-maregramme
```

### 2. Créer un environnement virtuel (Recommandé)
Il est fortement conseillé d'utiliser un environnement virtuel pour isoler les bibliothèques du projet et éviter les conflits de versions.

* **Sur Windows (Invite de commandes / CMD) :**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
  

* **Sur macOS/Linux :**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Installer les dépendances
Une fois votre environnement virtuel activé, installez l'ensemble des bibliothèques requises en exécutant la commande suivante :
  ```bash
  pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Lancer le programme
 
