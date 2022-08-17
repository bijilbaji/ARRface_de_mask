import cv2
import numpy as np
import face_recognition
import os

fpath = os.path.dirname(os.path.realpath(__file__))+"\\"
path = fpath + 'face\\'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print('Loaded names')
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Face Recgonission Loaded')

def facedetect(img):
    dt_name = ''
    unfc = 0
    # imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            dt_name = dt_name+' '+name
    #         y1,x2,y2,x1 = faceLoc
    #         y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    #         cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    #         cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
    #         cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),2)
        else:
            print("unknown face")
            unfc = unfc+1
    # cv2.imwrite(fpath+"/detimg.jpg",img)
    return dt_name, unfc
# cv2.imshow('Webcam',img)
# cv2.waitKey(0)
