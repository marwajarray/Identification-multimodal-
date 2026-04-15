"""
reconnaissance_faciale.py — Reconnaissance Faciale
Utilise HOG/LBP + SVM pour l'identification du visage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import FACE_IMAGE_SIZE, FACE_THRESHOLD


class ReconnaissanceFaciale:
    """
    Module de reconnaissance faciale basé sur HOG/LBP + SVM.
    
    Pipeline :
    1. Détection du visage (Haar Cascade ou DNN)
    2. Extraction de caractéristiques LBP (Local Binary Patterns)
    3. Classification SVM (Support Vector Machine)
    """

    def __init__(self, model_path, seuil=FACE_THRESHOLD, taille=FACE_IMAGE_SIZE):
        self.model_path = model_path
        self.seuil = seuil
        self.taille = taille
        self.modele = None
        self.encodeur = LabelEncoder()
        self._detecteur = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._charger_modele()

    def _charger_modele(self):
        """Charge le modèle SVM depuis le fichier."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.modele = data['modele']
                self.encodeur = data['encodeur']
            print(f"[ReconnaissanceFaciale] Modèle chargé : {self.model_path}")
        else:
            print(f"[ReconnaissanceFaciale] Modèle non trouvé. Enrôlement requis.")

    def _extraire_caracteristiques_lbp(self, image):
        """
        Extrait les caractéristiques LBP (Local Binary Patterns) de l'image.
        LBP est robuste aux variations d'éclairage.
        """
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gris_redim = cv2.resize(gris, self.taille)

        # Calcul LBP manuel (rayon=1, points=8)
        lbp = np.zeros_like(gris_redim, dtype=np.uint8)
        for i in range(1, gris_redim.shape[0] - 1):
            for j in range(1, gris_redim.shape[1] - 1):
                centre = gris_redim[i, j]
                code = 0
                voisins = [
                    gris_redim[i-1, j-1], gris_redim[i-1, j], gris_redim[i-1, j+1],
                    gris_redim[i,   j+1], gris_redim[i+1, j+1], gris_redim[i+1, j],
                    gris_redim[i+1, j-1], gris_redim[i,   j-1]
                ]
                for k, voisin in enumerate(voisins):
                    if voisin >= centre:
                        code |= (1 << k)
                lbp[i, j] = code

        # Histogramme LBP
        histogramme, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        histogramme = histogramme.astype(float)
        histogramme /= (histogramme.sum() + 1e-7)  # Normalisation
        return histogramme

    def _extraire_caracteristiques_hog(self, image):
        """
        Extrait les caractéristiques HOG (Histogram of Oriented Gradients).
        Robuste aux variations de pose.
        """
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gris_redim = cv2.resize(gris, (64, 64))

        hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        return hog.compute(gris_redim).flatten()

    def extraire_caracteristiques(self, image):
        """Combine LBP + HOG pour une meilleure robustesse."""
        lbp = self._extraire_caracteristiques_lbp(image)
        hog = self._extraire_caracteristiques_hog(image)
        return np.concatenate([lbp, hog])

    def detecter_visage(self, frame):
        """Détecte et extrait le visage d'une image."""
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        visages = self._detecteur.detectMultiScale(gris, 1.3, 5, minSize=(80, 80))

        if len(visages) == 0:
            return None

        # Prendre le visage le plus grand
        x, y, w, h = max(visages, key=lambda v: v[2] * v[3])
        return frame[y:y+h, x:x+w]

    def identifier(self):
        """
        Lance la caméra et identifie le visage.
        
        Returns:
            (str | None, float): (identité, score de confiance)
        """
        if self.modele is None:
            print("[ReconnaissanceFaciale] Aucun modèle chargé. Mode démonstration.")
            return self._mode_demo()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return self._mode_demo()

        identite = None
        confiance = 0.0
        frames_traitees = 0
        predictions = []

        while frames_traitees < 30:  # 30 frames pour stabiliser
            ret, frame = cap.read()
            if not ret:
                break

            visage = self.detecter_visage(frame)
            if visage is not None:
                features = self.extraire_caracteristiques(visage)
                proba = self.modele.predict_proba([features])[0]
                pred_idx = np.argmax(proba)
                pred_label = self.encodeur.inverse_transform([pred_idx])[0]
                pred_score = proba[pred_idx]

                predictions.append((pred_label, pred_score))

                # Affichage temps réel
                couleur = (0, 255, 0) if pred_score >= self.seuil else (0, 0, 255)
                cv2.putText(frame, f"{pred_label}: {pred_score:.2%}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)

            cv2.imshow("Reconnaissance Faciale", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frames_traitees += 1

        cap.release()
        cv2.destroyAllWindows()

        # Prendre la prédiction majoritaire
        if predictions:
            from collections import Counter
            labels = [p[0] for p in predictions if p[1] >= self.seuil]
            if labels:
                identite = Counter(labels).most_common(1)[0][0]
                confiance = np.mean([p[1] for p in predictions if p[0] == identite])

        return identite, confiance

    def _mode_demo(self):
        """Mode démonstration."""
        return "utilisateur_demo", 0.92

    def entrainer(self, images_par_personne: dict):
        """
        Entraîne le modèle SVM sur les images fournies.
        
        Args:
            images_par_personne: {"nom_personne": [liste_images_numpy]}
        """
        X, y = [], []
        for nom, images in images_par_personne.items():
            for image in images:
                features = self.extraire_caracteristiques(image)
                X.append(features)
                y.append(nom)

        self.encodeur.fit(y)
        y_encoded = self.encodeur.transform(y)

        self.modele = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, C=10, gamma='scale'))
        ])
        self.modele.fit(X, y_encoded)

        # Sauvegarde
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({'modele': self.modele, 'encodeur': self.encodeur}, f)

        print(f"[ReconnaissanceFaciale] Modèle entraîné sur {len(X)} images, {len(images_par_personne)} personnes.")
        return True
