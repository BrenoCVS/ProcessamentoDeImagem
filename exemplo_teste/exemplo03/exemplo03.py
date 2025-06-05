import cv2

face_cascade = cv2.CascadeClassifier('../../haarcascade_frontalface_default.xml')


if face_cascade.empty():
    print("Erro ao carregar o classificador Haar Cascade")
    exit()


img = cv2.imread('image/foto.jpeg')

if img is None:
    print("Erro ao carregar a imagem")
    exit()

largura = int(img.shape[1] * 2)
altura = int(img.shape[0] * 2)
dimensao = (largura, altura)

img_redimensionada = cv2.resize(img, dimensao, interpolation=cv2.INTER_CUBIC)

img_suavizada = cv2.bilateralFilter(img_redimensionada, d=9, sigmaColor=75, sigmaSpace=75)

cv2.imshow('Imagem', img_suavizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for(x, y, w, h) in faces:
    cv2.rectangle(img_suavizada, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Rostos Detectados', img_suavizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"{len(faces)} rosto(s) detectado(s)")

for i, (x, y, w, h) in enumerate(faces, start=1):
    rosto = img_suavizada[y:y+h, x:x+w]
    rosto_cinza = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)
    nome_arquivo = f"image/faces/rosto_{i}.jpg"
    cv2.imwrite(nome_arquivo, rosto_cinza)
    print(f"Salvo: {nome_arquivo}")


