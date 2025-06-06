#Le e processa uma imagem procurando rostos

import cv2

face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

img = cv2.imread('image/foto.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Rostos Detectados', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
