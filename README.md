# Numérisation Automatique de Marégrammes Papier

Ce projet a été réalisé par **Valentin GEGADEN** dans le cadre d'un stage de fin de Licence (L3) à **La Rochelle Université**, au sein des laboratoires **MIA** (Mathématiques, Image et Applications) et **LIENSs** (Littoral, ENvironnement et Sociétés), en collaboration avec les données historiques du **Shom** (Service hydrographique et océanographique de la Marine).

L'objectif principal est de développer un outil/algorithme en Python capable d'automatiser la numérisation des marégrammes papier scannés (fichiers `.tif`) issus du port de Douala (Cameroun), afin de pallier les limites du logiciel semi-automatique actuel, NUNIEAU.

---

## 🚀 Fonctionnalités du Projet

L'algorithme découpe le processus de traitement d'image en plusieurs étapes clés :
* **Identification de la date** : Une interaction console avec l'utilisateur permet de définir les dates de pose et de retrait afin de calculer le nombre théorique de courbes à extraire.
* **Extraction de la grille** : Détection géométrique des lignes horizontales et verticales via leurs intersections pour recadrer, rogner et redresser (transformation affine) le marégramme sans altérer la résolution d'origine.
* **Extraction des courbes** : Segmentation avancée par couleur en utilisant l'espace HSV (Teinte, Saturation, Luminosité) et l'analyse d'histogramme pour isoler automatiquement la teinte dominante de la courbe (bleu foncé/noir) du fond blanc et de la grille.
* **Squelettisation** : Application de techniques de morphologie mathématique pour réduire l'épaisseur des courbes à un seul pixel et faciliter l'association temporelle et de hauteur d'eau.
* **Suivi des courbes & Gestion des croisements** : Détection des croisements et prédiction de la tendance physique à suivre grâce à un calcul de régression linéaire basé sur l'inertie naturelle de la courbe.
* **Suivi du jour suivant** : Transition automatique d'une courbe vers la colonne X=0 du jour suivant par continuité physique de la marée, en exploitant les arbres à K dimensions (KD-Trees) pour la recherche des voisins les plus proches.

---

## 🛠️ Installation et Configuration

Suivez ces étapes pour installer le projet et configurer votre environnement de développement.

### Prérequis
Assurez-vous d'avoir **Python 3.9+** installé sur votre machine.

### 1. Cloner le projet
```bash
git clone [https://github.com/votre-utilisateur/numerisation-maregrammes.git](https://github.com/votre-utilisateur/numerisation-maregrammes.git)
cd numerisation-maregrammes
