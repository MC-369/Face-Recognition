import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('images\elon train.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgElonTest = face_recognition.load_image_file('images\Tesla-Elon-Musk-India-1024x576.jpg')
imgElonTest = cv2.cvtColor(imgElonTest,cv2.COLOR_BGR2RGB)
#
# imgBillgateTest = face_recognition.load_image_file('images/billgates test.jpg')
# imgBillgateTest = cv2.cvtColor(imgBillgateTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgElonTest)[0]
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeElon],encodeElonTest)
faceDis = face_recognition.face_distance([encodeElon],encodeElonTest)
print(result,faceDis)  ## shroter the distance good matches between the faces
cv2.putText(imgElonTest,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Musk Test image',imgElonTest)
# cv2.imshow('Bill Gate Test image',imgBillgateTest)
cv2.waitKey(0)

