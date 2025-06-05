#Leitura e exibição de uma imagem

import cv2

imagem = cv2.imread('image/Breno.jpg')

cv2.imshow('Minha imagem', imagem)

cv2.waitKey(0)

cv2.destroyAllWindows()
