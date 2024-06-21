import cv2
import sys
from mtcnn import MTCNN
from flou import flou

# Charger l'image
nom = input("Entrer le nom de l'image : ")

image = cv2.imread(nom)

if image is None:
    print("Image introuvable !")
    sys.exit()

image_width = image.shape[1]
image_height = image.shape[0]

# Initialiser le détecteur de visages
detector = MTCNN(None,13,None,0.95)
face_cascade = cv2.CascadeClassifier('lbpcascade_profileface.xml')


result = detector.detect_faces(image)
faces = face_cascade.detectMultiScale(image,2.3,0)
image = cv2.flip(image, 1)
faces_flipped = face_cascade.detectMultiScale(image,2.3,0)
image = cv2.flip(image, 1)

# Convertir les résultats de MTCNN en une forme compatible avec les autres résultats
result = [(x, y, w, h) for res in result for x, y, w, h in [res['box']]]

faces = list(faces)
faces += [(x, y, w, h) for (x, y, w, h) in faces_flipped]
faces += result

print(f"{len(faces)} visages uniques détectés dans l'image.")

# Appliquer le flou sur chaque visage détecté
for (x, y, width, height) in faces :
	for i in range(x, (x + width-1),50) :
		for j in range(y, (y + height-1),50) :
			if(i<image_width and j<image_height) : image = flou(image,i,j)

print ("C'est tout bon!")
cv2.imwrite("new.jpg", image)
