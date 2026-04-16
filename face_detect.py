import cv2
import tensorflow as tf
import numpy as np

# Load gender model
model = tf.keras.models.load_model("model/gender_model.keras")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 🔹 Crop face
        face = frame[y:y+h, x:x+w]

        # 🔹 Resize to model input
        face = cv2.resize(face, (64, 64))

        # 🔹 Normalize
        face = face / 255.0

        # 🔹 Reshape
        face = np.reshape(face, (1, 64, 64, 3))

        # 🔹 Predict
        prediction = model.predict(face, verbose=0)

        label = "Female" if prediction[0][0] > 0.5 else "Male"

        # 🔹 Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 🔹 Show label
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("AIVA Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()