import cv2 as cv
import numpy as np
import face_recognition as fr
import os
#низкая частота кадров, поскольку распознование лица происходит cо сменой каждого кадра
path = 'images'
img = []
classNames = []
try:
    mlis = os.listdir(path)
except:
    raise ValueError("папка с фото 'images' не найдена")
#ввод изображений из директории path ('images')
for name in mlis:
    curImg = cv.imread(f'{path}/{name}')
    img.append(curImg)
    classNames.append(name.split(".")[0])
    print(classNames[-1])
#декодирование изображений (исходных)
def findEncodings(images):
    encodelist = []
    print("|", end="")
    for i in range(len(images)):
        print("=", end="")
    print("|")
    print("|", end="")
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tmp = fr.face_encodings(img)[0]
        encodelist.append(tmp)
        print("#", end="")
    print("|")
    return encodelist
referece = findEncodings(img)# референсы, на основе их ищем в новых изображениях лица

cap = cv.VideoCapture(0)
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    #форматируем frame для передачи в декодер
    IMG = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    IMG = cv.cvtColor(IMG, cv.COLOR_BGR2RGB)
    #gray = cv.resize(gray, (gray.shape[1], gray.shape[0] // 3))  # размер
    face_position = fr.face_locations(IMG)
    encode_frame = fr.face_encodings(IMG, face_position)
    #для frame ищем совпадения с фото из reference сравнивая функцией compare_faces
    for encode_face, face_location in zip(encode_frame, face_position):
        match = fr.compare_faces(referece, encode_face)
        faceDis = fr.face_distance(referece, encode_face)
        matchIndex = np.argmin(faceDis)
        if match[matchIndex]:
            #рисование прямоугольника вокруг головы
            name = classNames[matchIndex]
            print(name)
            """y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1 * 4 - 10, x2 * 4 +10, y2 * 4 + 10, x1 * 4 - 10
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(frame, (x1, y2), (x2, y2+35), (0, 255, 0), cv.FILLED)"""
            #надпись
            #cv.putText(frame, name, (x1 + 6, y2 + 20), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        #print(faceDis)
    #cv.imshow('vid', frame)