# Inspired by a TechWithTim Video
# https://youtu.be/D5xqcGk6LEc




import face_recognition as faceRec
import os
import cv2
import face_recognition
import numpy
from time import sleep


# Looks through the ./faces folder and encodes all the images in that folder
# returns the directory of the name and the image encoded


def encodeFaces():


   encoded = {}


   for directoryPath , directoryNames, faceNames in os.walk("./faces"):


       for f in faceNames:


           if f.endswith(".jpg") or f.endswith(".png"):
               face = faceRec.load_image_file("faces/" + f)
               faceEncoding = faceRec.face_encodings(face)[0]
               encoded[f.split(".")[0]] = faceEncoding




   return encoded


# encodes a face given the name of the file
# returns the encoding of the face in that file




def unknownImageEncoding(image):


   face = faceRec.load_image_file("faces/" + image)
   faceEncoding = faceRec.face_encodings(face)[0]


   return faceEncoding


# finds all the faces in the given image and labels them if it recognizes them


def classifyFace(image):


   face = encodeFaces()
   facesEncoded = list(face.values())
   knownFaces = list(face.keys())


   img = cv2.imread(image, 1)
   img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)


   faceLocations = face_recognition.face_locations(img)
   unknownFaceEncodings = face_recognition.face_encodings(img)


   faceNames = []


   for faceEncoding in unknownFaceEncodings:


       # see if the face matches the known faces


       matches = face_recognition.compare_faces(facesEncoded, faceEncoding)
       name = 'unknown'


       # uses the known face and calculates the smallest distance to the new face


       faceDistances = face_recognition.face_distance(facesEncoded, faceEncoding)
       bestMatchIndex = numpy.argmin(faceDistances)


       if matches[bestMatchIndex]:
           name = knownFaces[bestMatchIndex]
       faceNames.append(name)


       for (top, right, bottom, left), name in zip(faceLocations, faceNames):


           # draws a rectangle around the face
           cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)


           # draws a label with name at the bottom for the face


           cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
           font = cv2.FONT_HERSHEY_DUPLEX
           cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2)




   # displays the image in new window and its results
   while True:


       cv2.imshow("Video", img)


       if cv2.waitKey(1) & 0xFF == ord('q'):
           return faceNames


print(classifyFace("test.jpg"))
