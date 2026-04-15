import cv2
import os
import time

# 🔥 demander nom utilisateur
user_name = input("Entrez le nom de l'utilisateur : ").strip()

dataset_path = f"dataset/{user_name}"
os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

count = 0
capturing = False
last_capture_time = 0

print("Appuie sur 's' pour commencer")
print("Appuie sur 'q' pour quitter")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Erreur caméra")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

        if capturing:
            current_time = time.time()

            # 🔥 capture toutes les 1 seconde
            if current_time - last_capture_time > 1:
                count += 1
                face = gray[y:y+h, x:x+w]

                file_name = f"{dataset_path}/img_{count}.jpg"
                cv2.imwrite(file_name, face)

                last_capture_time = current_time

    cv2.putText(frame, f"User: {user_name}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"Images: {count}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture Dataset", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        capturing = True
        print(f"📸 Capture pour {user_name}...")

    if key == ord('q') or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Dataset créé pour {user_name}")