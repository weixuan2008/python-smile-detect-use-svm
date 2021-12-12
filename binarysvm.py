from dataprocess import dataprocess
from dataprocess import getmyself
import numpy as np 
import matplotlib.pyplot as plt 
import face_recognition
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data = dataprocess()
data = data.to_numpy()
#print(data)
x, y = np.split(data, (136,), axis=1)
x = x[:, :]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
# 训练svm分类器
clf = SVC(C=0.8, kernel='rbf', gamma=1, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
#计算svc分类器的准确率
print(clf.score(x_train, y_train))  # 精度
y_hat = clf.predict(x_train)
#print(x_train)
#print(y_hat)
print(clf.score(x_test, y_test))
y_hat = clf.predict(x_test)


#用自己的人脸数据做检测
#myself=getmyself()
#myself=myself.to_numpy()
#print(myself)
#AmISmile=clf.predict(myself)
#print(AmISmile)

#视频检测
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import time
import dlib
video_capture = cv2.VideoCapture(0)
j=0
detector = dlib.get_frontal_face_detector()
predictor_path = "./train_dir/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
faces=[]
while j<100:
    
     # 抓取一帧视频
    ret, frame = video_capture.read()

    picture = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    img=np.array(picture)
    dets = detector(img, 1)
    face=[]
    for k, d in enumerate(dets):
        # 得到的地标/用于面部在框d的部分。
        shape = predictor(img, d)
        rect_w=shape.rect.width()
        rect_h=shape.rect.height()
        top=shape.rect.top()
        left=shape.rect.left()
        for i in range(shape.num_parts):
            #print("{},{},".format((shape.part(i).x-left)/rect_w,(shape.part(i).y-top)/rect_h))
            face=np.append(face,(shape.part(i).x-left)/rect_w)
            face=np.append(face,(shape.part(i).y-top)/rect_h)
    #img = dlib.convert_image(picture)
    print(type(face))
    print(len(face))
    if(len(face)!=0):
        faces.append(face)
        print(faces)
        j+=1
        smile=clf.predict(faces)
        issmile=""
        if(smile[0]==1):
            print("smile")
            issmile="smile"
        else:
            print("nosmile")
            issmile="nosmile"
        faces.clear()
        #cv2.imshow('face',cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        
        # 将图像从 BGR 颜色（OpenCV 使用的）转换为 RGB 颜色（face_recognition 使用的）
        rgb_frame = frame[:, :, ::-1]

        
        # 找到视频帧中的所有人脸和人脸编码
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # 遍历这一帧视频中的每个人脸
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 在脸部周围画一个框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 在人脸下方画一个带有名字的标签
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, issmile, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    

        # 显示结果图像
        cv2.imshow('笑脸检测', frame)

        # 按键盘上的“q”退出！
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #cv2.waitKey(0)
        #time.sleep(1)
    #time.sleep(2)
    #img = dlib.load_rgb_image(f)
    


# 释放网络摄像头的句柄
video_capture.release()
cv2.destroyAllWindows()
