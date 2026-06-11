# Numérisation Automatique de Marégrammes Papier

[cite_start]Ce projet a été réalisé par **Valentin GEGADEN** [cite: 4] [cite_start]dans le cadre d'un stage de fin de Licence (L3) [cite: 388] [cite_start]à **La Rochelle Université** [cite: 1, 12][cite_start], au sein des laboratoires **MIA** (Mathématiques, Image et Applications) [cite: 12, 13] [cite_start]et **LIENSs** (Littoral, ENvironnement et Sociétés) [cite: 12, 18][cite_start], en collaboration avec les données historiques du **Shom** (Service hydrographique et océanographique de la Marine)[cite: 12, 23].

[cite_start]L'objectif principal est de développer un outil/algorithme en Python [cite: 68, 127] [cite_start]capable d'automatiser la numérisation des marégrammes papier scannés (fichiers `.tif`) [cite: 69, 73] [cite_start]issus du port de Douala (Cameroun) [cite: 28][cite_start], afin de pallier les limites du logiciel semi-automatique actuel, NUNIEAU[cite: 60, 63, 64].

---

## 🚀 Fonctionnalités du Projet

[cite_start]L'algorithme découpe le processus de traitement d'image en plusieurs étapes clés[cite: 116]:
* [cite_start]**Identification de la date** : Une interaction console avec l'utilisateur permet de définir les dates de pose et de retrait afin de calculer le nombre théorique de courbes à extraire[cite: 130, 138, 141].
* [cite_start]**Extraction de la grille** : Détection géométrique des lignes horizontales et verticales via leurs intersections pour recadrer, rogner et redresser (transformation affine) le marégramme sans altérer la résolution d'origine[cite: 159, 161, 165, 169].
* [cite_start]**Extraction des courbes** : Segmentation avancée par couleur en utilisant l'espace HSV et l'analyse d'histogramme pour isoler automatiquement la teinte dominante de la courbe (bleu foncé/noir) du fond blanc et de la grille[cite: 184, 186, 195, 199].
* [cite_start]**Squelettisation** : Application de techniques de morphologie mathématique pour réduire l'épaisseur des courbes à un seul pixel et faciliter l'association temporelle et de hauteur d'eau[cite: 203, 204, 205].
* [cite_start]**Suivi des courbes & Gestion des croisements** : Détection des croisements et prédiction de la tendance physique à suivre grâce à un calcul de régression linéaire basé sur l'inertie naturelle de la courbe[cite: 249, 250, 253, 257].
* [cite_start]**Suivi du jour suivant** : Transition automatique d'une courbe vers la colonne $X=0$ du jour suivant par continuité physique de la marée, en exploitant les arbres à K dimensions (KD-Trees) pour la recherche des voisins les plus proches[cite: 280, 281, 388].

---

## 🛠️ Installation et Configuration

Suivez ces étapes pour installer le projet et configurer votre environnement de développement.

### Prérequis
Assurez-vous d'avoir **Python 3.9+** installé sur votre machine.

### 1. Cloner le projet
```bash
git clone [https://github.com/votre-utilisateur/numerisation-maregrammes.git](https://github.com/votre-utilisateur/numerisation-maregrammes.git)
cd numerisation-maregrammes
