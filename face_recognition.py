import cv2
import matplotlib.pyplot as plt

imag3=cv2.imread('F:\IMG-20180506-WA0125.jpg') 
cascade=cv2.CascadeClassifier('E:\OpenCV\haar-cascade-files-master\haarcascade_frontalface_default.xml')

def rgb_convert(imag3):
    return cv2.cvtColor(imag3,cv2.COLOR_BGR2RGB)

def detect_faces(self, imag3, scaleFactor = 1.1):
    image_copy = imag3.copy() #copy to prevent any changes
    
    faces1 = cascade.detectMultiScale(image_copy, scaleFactor=scaleFactor, minNeighbors=5)  # Applying the haar classifier to detect faces
    print('faces found',len(faces1))

    for (x, y, w, h) in faces1:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 4) #image,Vertex of the rectangle1,Vertex of the rectangle_opposite to rectangle1,color,thickness
    return image_copy

faces = detect_faces(cascade, imag3)
plt.imshow(rgb_convert(faces))
plt.title('FACE DETECTION')
plt.show()
