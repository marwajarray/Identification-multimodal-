"""
reconnaissance_empreinte.py — Reconnaissance d'Empreinte Digitale
Utilise l'extraction de minuties et le matching pour identifier l'empreinte.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
from config import FINGERPRINT_THRESHOLD, FINGERPRINT_SIZE


class ReconnaissanceEmpreinte:
    """
    Module de reconnaissance d'empreinte digitale.
    
    Pipeline :
    1. Prétraitement (normalisation, binarisation, amincissement)
    2. Extraction des minuties (terminaisons et bifurcations)
    3. Matching par distance euclidienne entre sets de minuties
    """

    def __init__(self, seuil=FINGERPRINT_THRESHOLD, taille=FINGERPRINT_SIZE):
        self.seuil = seuil
        self.taille = taille

    def _pretraiter(self, image):
        """Prétraite l'image d'empreinte pour l'extraction de minuties."""
        # Redimensionnement
        if len(image.shape) == 3:
            gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gris = image.copy()

        redim = cv2.resize(gris, self.taille)

        # Normalisation CLAHE (améliore le contraste localement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalise = clahe.apply(redim)

        # Binarisation avec seuillage d'Otsu
        _, binaire = cv2.threshold(normalise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Inversion si nécessaire (crêtes en noir sur fond blanc)
        if np.sum(binaire == 0) < np.sum(binaire == 255):
            binaire = cv2.bitwise_not(binaire)

        # Amincissement des crêtes (skeletonization)
        squelette = self._squelettiser(binaire)
        return squelette

    def _squelettiser(self, binaire):
        """Réduit les crêtes à 1 pixel d'épaisseur (Zhang-Suen)."""
        squelette = np.zeros_like(binaire)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binaire.copy()

        while True:
            erode = cv2.erode(temp, element)
            ouverte = cv2.dilate(erode, element)
            soustraction = cv2.subtract(temp, ouverte)
            squelette = cv2.bitwise_or(squelette, soustraction)
            temp = erode.copy()
            if cv2.countNonZero(temp) == 0:
                break

        return squelette

    def extraire_minuties(self, image):
        """
        Extrait les minuties (terminaisons et bifurcations) de l'empreinte.
        
        Returns:
            list: Liste de tuples (x, y, type, angle)
                  type: 'T' = terminaison, 'B' = bifurcation
        """
        squelette = self._pretraiter(image)
        minuties = []
        h, w = squelette.shape

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if squelette[i, j] == 0:  # Pixel de crête
                    # Voisinage 3x3
                    voisins = [
                        squelette[i-1, j-1], squelette[i-1, j], squelette[i-1, j+1],
                        squelette[i,   j+1], squelette[i+1, j+1], squelette[i+1, j],
                        squelette[i+1, j-1], squelette[i,   j-1]
                    ]
                    # Nombre de pixels noirs (crêtes)
                    nb_voisins_cretes = sum(1 for v in voisins if v == 0)

                    # Nombre de transitions 0→255 (crossing number)
                    voisins_ext = voisins + [voisins[0]]
                    transitions = sum(
                        1 for k in range(len(voisins))
                        if voisins_ext[k] == 0 and voisins_ext[k+1] == 255
                    )

                    # Terminaison : 1 voisin crête, 1 transition
                    if transitions == 1 and nb_voisins_cretes == 1:
                        angle = np.degrees(np.arctan2(i - h//2, j - w//2))
                        minuties.append((j, i, 'T', angle))

                    # Bifurcation : 3 voisins crêtes, 3 transitions
                    elif transitions == 3 and nb_voisins_cretes == 3:
                        angle = np.degrees(np.arctan2(i - h//2, j - w//2))
                        minuties.append((j, i, 'B', angle))

        return minuties

    def matcher(self, minuties1, minuties2):
        """
        Compare deux ensembles de minuties.
        
        Returns:
            (int, float): (nb_correspondances, score_similarite)
        """
        if not minuties1 or not minuties2:
            return 0, 0.0

        correspondances = 0
        tolerance_spatiale = 15  # pixels
        tolerance_angle = 15     # degrés

        for m1 in minuties1:
            x1, y1, type1, angle1 = m1
            for m2 in minuties2:
                x2, y2, type2, angle2 = m2

                if type1 != type2:
                    continue

                dist_spatiale = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                dist_angle = abs(angle1 - angle2) % 360
                dist_angle = min(dist_angle, 360 - dist_angle)

                if dist_spatiale <= tolerance_spatiale and dist_angle <= tolerance_angle:
                    correspondances += 1
                    break

        total = min(len(minuties1), len(minuties2))
        score = correspondances / total if total > 0 else 0.0
        return correspondances, score

    def authentifier(self, image_test, gabarit_reference):
        """
        Authentifie une empreinte par rapport à un gabarit de référence.
        
        Args:
            image_test: Image numpy de l'empreinte à tester
            gabarit_reference: Minuties de référence (liste)
            
        Returns:
            (bool, int, float): (authentifié, nb_correspondances, score)
        """
        minuties_test = self.extraire_minuties(image_test)
        nb_corr, score = self.matcher(minuties_test, gabarit_reference)

        authentifie = nb_corr >= self.seuil
        return authentifie, nb_corr, score

    def identifier(self):
        """Mode démonstration (sans capteur réel)."""
        print("[ReconnaissanceEmpreinte] Mode démonstration activé")
        # Simuler une empreinte générée aléatoirement
        image_simulee = np.random.randint(0, 256, (*self.taille, 3), dtype=np.uint8)
        minuties = self.extraire_minuties(image_simulee)
        return True, len(minuties), 0.87

    def visualiser_minuties(self, image, minuties):
        """
        Affiche les minuties détectées sur l'image.
        
        Returns:
            numpy.ndarray: Image annotée
        """
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

        for (x, y, type_min, angle) in minuties:
            couleur = (0, 255, 0) if type_min == 'T' else (0, 0, 255)
            cv2.circle(img_color, (x, y), 4, couleur, -1)

            # Dessiner la direction
            dx = int(10 * np.cos(np.radians(angle)))
            dy = int(10 * np.sin(np.radians(angle)))
            cv2.arrowedLine(img_color, (x, y), (x+dx, y+dy), couleur, 1, tipLength=0.3)

        # Légende
        cv2.putText(img_color, "Vert = Terminaison", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img_color, "Rouge = Bifurcation", (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img_color, f"Total: {len(minuties)} minuties", (5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img_color
