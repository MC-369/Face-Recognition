import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'imgs'
imgList = []
classNames = []
mYList = os.listdir(path)
print(mYList)

for cl in mYList:
    curImg = cv2.imread(f'{path}/{cl}')
    imgList.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findencodings(imgList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')




encodeListKnown = findencodings(imgList)
print('encoding completed')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read(0)
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace, faceLOc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLOc
            # y1, x2, y2, x1 = y1,x2*8,y2*8,x1*8
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            markAttendence(name)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)
