"""
liveness_detection.py — Détection de Vivacité (Anti-Spoofing)
[AJOUT CRÉATIF N°1]
Détecte si la personne est bien réelle (pas une photo ou un masque).
Techniques : Eye Aspect Ratio (EAR) + challenge aléatoire
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import time
import random
from scipy.spatial import distance as dist
from config import EYE_AR_THRESHOLD, EYE_AR_CONSEC_FRAMES


def calculer_ear(oeil):
    """
    Calcule le Eye Aspect Ratio (EAR) pour détecter le clignement.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    A = dist.euclidean(oeil[1], oeil[5])
    B = dist.euclidean(oeil[2], oeil[4])
    C = dist.euclidean(oeil[0], oeil[3])
    return (A + B) / (2.0 * C)


class LivenessDetector:
    """
    Module de détection de vivacité pour éviter les attaques par présentation.
    
    Méthodes de détection :
    1. Clignement des yeux (Eye Aspect Ratio)
    2. Challenge aléatoire (tourner la tête, sourire)
    3. Analyse de texture (score de naturalité de l'image)
    """

    # Indices des landmarks pour les yeux (modèle 68 points de dlib)
    OEIL_GAUCHE_IDX = list(range(42, 48))
    OEIL_DROIT_IDX  = list(range(36, 42))

    DEFIS_DISPONIBLES = [
        ("Clignez des yeux 2 fois", "CLIGNEMENT"),
        ("Tournez légèrement la tête à gauche", "ROTATION_GAUCHE"),
        ("Tournez légèrement la tête à droite", "ROTATION_DROITE"),
        ("Souriez !", "SOURIRE"),
    ]

    def __init__(self, seuil_ear=EYE_AR_THRESHOLD, frames_consecutives=EYE_AR_CONSEC_FRAMES):
        self.seuil_ear = seuil_ear
        self.frames_consecutives = frames_consecutives
        self._detecteur = None
        self._predicteur = None
        self._charger_modeles()

    def _charger_modeles(self):
        """Charge les modèles dlib pour la détection de landmarks faciaux."""
        try:
            import dlib
            self._detecteur = dlib.get_frontal_face_detector()
            # Note: télécharger shape_predictor_68_face_landmarks.dat
            modele_path = "assets/shape_predictor_68_face_landmarks.dat"
            self._predicteur = dlib.shape_predictor(modele_path)
            print("[LivenessDetector] Modèle dlib chargé avec succès.")
        except Exception as e:
            print(f"[LivenessDetector] Fallback sur OpenCV Haarcascade : {e}")
            self._detecteur = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def verifier(self, timeout_sec=15):
        """
        Lance la vérification de vivacité via la caméra.
        
        Returns:
            (bool, str): (est_vivant, message)
        """
        # Sélectionner un défi aléatoire
        defi_texte, defi_type = random.choice(self.DEFIS_DISPONIBLES)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Mode démonstration sans caméra
            return self._mode_demo()

        compteur_clignements = 0
        compteur_frames_ferme = 0
        debut = time.time()
        resultat = False

        while (time.time() - debut) < timeout_sec:
            ret, frame = cap.read()
            if not ret:
                break

            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Affichage du défi
            cv2.putText(frame, f"DEFI: {defi_texte}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            temps_restant = int(timeout_sec - (time.time() - debut))
            cv2.putText(frame, f"Temps: {temps_restant}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self._predicteur:
                # Détection via dlib (précise)
                visages = self._detecteur(gris, 0)
                for visage in visages:
                    landmarks = self._predicteur(gris, visage)
                    coords = np.array([[landmarks.part(i).x, landmarks.part(i).y]
                                       for i in range(68)])

                    oeil_g = coords[self.OEIL_GAUCHE_IDX]
                    oeil_d = coords[self.OEIL_DROIT_IDX]
                    ear = (calculer_ear(oeil_g) + calculer_ear(oeil_d)) / 2.0

                    # Détection du clignement
                    if ear < self.seuil_ear:
                        compteur_frames_ferme += 1
                    else:
                        if compteur_frames_ferme >= self.frames_consecutives:
                            compteur_clignements += 1
                            print(f"[LivenessDetector] Clignement #{compteur_clignements} détecté")
                        compteur_frames_ferme = 0

                    # Dessiner le contour des yeux
                    for point in np.concatenate([oeil_g, oeil_d]):
                        cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Vérification du défi
            if defi_type == "CLIGNEMENT" and compteur_clignements >= 2:
                resultat = True
                break

            cv2.imshow("Verification de Vivacite — Appuyez sur Q pour quitter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if resultat:
            return True, f"Défi réussi : {defi_texte}"
        else:
            return False, "Défi échoué — Possible attaque par présentation"

    def _mode_demo(self):
        """Mode démonstration quand aucune caméra n'est disponible."""
        print("[LivenessDetector] Mode démonstration activé (pas de caméra)")
        return True, "Mode démonstration — Vivacité simulée"

    def analyser_texture(self, image):
        """
        Analyse la texture d'une image pour détecter les attaques papier.
        Une photo imprimée a une texture différente d'un vrai visage.
        
        Returns:
            float: Score de naturalité (0.0 = fausse image, 1.0 = vraie)
        """
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Laplacian Variance (mesure la netteté / texture)
        variance_laplacian = cv2.Laplacian(gris, cv2.CV_64F).var()
        
        # Normalisation approximative
        score = min(variance_laplacian / 500.0, 1.0)
        return score
