import cv2
import pickle
import os

# 🔥 Vérifier si modèle existe
if not os.path.exists("trainer/model.yml"):
    print("❌ Modèle non trouvé ! Lance train.py d'abord")
    exit()

# 🔥 Charger modèle
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/model.yml")

# 🔥 Charger labels automatiquement
if not os.path.exists("trainer/labels.pkl"):
    print("❌ labels.pkl non trouvé ! Lance train.py")
    exit()

with open("trainer/labels.pkl", "rb") as f:
    label_map = pickle.load(f)

print("👥 Utilisateurs chargés :", list(label_map.values()))

# 🔥 Détection visage
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Problème caméra")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 🔴 Aucun visage
    if len(faces) == 0:
        cv2.putText(frame, "Aucun visage", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        # 🔥 Sécurité : éviter crash si label inconnu
        if label in label_map:
            name = label_map[label]
        else:
            name = "Inconnu"

        # 🔥 Décision
        if confidence < 60:
            text = f"{name} ({round(confidence,2)})"
            color = (0, 255, 0)
        else:
            text = "Inconnu"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()