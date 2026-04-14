import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # get frames from video
    ret, frame = cap.read()

    # break if video fails
    if not ret:
        break

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # draw rectangle on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # show video
    cv2.imshow("AIVA Face Detection", frame)

    # wait for key press
    key = cv2.waitKey(25) & 0xFF

    # ESC to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()