import cv2
import os
import numpy as np
import pickle

dataset_path = "dataset"

faces = []
labels = []

label_id = 0
label_map = {}

# 🔥 Vérifier si dataset existe
if not os.path.exists(dataset_path):
    print("❌ Le dossier dataset n'existe pas !")
    exit()

# 🔥 trier les dossiers pour garder ordre stable
for person_name in sorted(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"📁 Chargement de : {person_name}")

    label_map[label_id] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(label_id)

    label_id += 1

# 🔥 Vérifier si données existent
if len(faces) == 0:
    print("❌ Aucune image trouvée dans dataset !")
    exit()

labels = np.array(labels)

# 🔥 Création modèle
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 🔥 Entraînement
recognizer.train(faces, labels)

# 🔥 Sauvegarde modèle
recognizer.save("trainer/model.yml")

# 🔥 Sauvegarde des labels
with open("trainer/labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Modèle entraîné avec succès !")
print(f"👥 Utilisateurs : {list(label_map.values())}")