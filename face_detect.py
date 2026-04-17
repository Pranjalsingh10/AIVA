import cv2
import tensorflow as tf
import numpy as np

# Load models
gender_model = tf.keras.models.load_model("model/gender_model.keras")
age_model = tf.keras.models.load_model("model/age_model.keras")

# Age labels (same as training)
age_groups = ["Young", "Adult", "Old"]
age_ranges = ["0-17 yr", "18-39 yr", "40+ yr"]

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
        # Crop face
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 3))

        # Gender Prediction
        gender_pred = gender_model.predict(face, verbose=0)
        gender = "Female" if gender_pred[0][0] > 0.5 else "Male"

        # Age Prediction
        age_pred = age_model.predict(face, verbose=0)
        age_class = np.argmax(age_pred)

        age_group = age_groups[age_class]
        age_range = age_ranges[age_class]

        # Final Label
        label = f"{gender}, {age_group}, {age_range}"

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put text
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("AIVA Age + Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()