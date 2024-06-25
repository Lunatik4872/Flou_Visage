import cv2
import sys
from mtcnn import MTCNN
from blur import blur

# Charger l'image
name = input("Enter the name of the image (name.jpg || name.png || [...]): ")

image = cv2.imread(name)

if image is None:
    print("Image not found !")
    sys.exit()

image_width = image.shape[1]
image_height = image.shape[0]

# Initialise face detectors
detector = MTCNN(None,13,None,0.95)
face_cascade = cv2.CascadeClassifier('lbpcascade_profileface.xml')


result = detector.detect_faces(image)
faces = face_cascade.detectMultiScale(image,2.3,0)
# Image rotation for face detection in profile
image = cv2.flip(image, 1)
faces_flipped = face_cascade.detectMultiScale(image,2.3,0)
image = cv2.flip(image, 1)

# Convert MTCNN results into a form compatible with other results
result = [(x, y, w, h) for res in result for x, y, w, h in [res['box']]]

faces = list(faces)
faces += [(x, y, w, h) for (x, y, w, h) in faces_flipped]
faces += result

print(f"{len(faces)} faces detected in the image.")

# Apply the blur to each face detected
for (x, y, width, height) in faces :
	for i in range(x, (x + width-1),50) :
		for j in range(y, (y + height-1),50) :
			if(i<image_width and j<image_height) : image = blur(image,i,j)

print("end of treatment !")
cv2.imwrite("new"+name+".jpg", image)
