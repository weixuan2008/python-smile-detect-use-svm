import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import time

# 这是一个超级简单（但速度较慢）的示例，可在网络摄像头的实时视频上运行面部识别。
# 还有第二个例子，它稍微复杂一点，但运行速度更快。

# 请注意：此示例需要安装 OpenCV（`cv2` 库），仅用于从您的网络摄像头读取数据。
# OpenCV *不需要*使用face_recognition库。只有当你想运行它时才需要它
# 具体演示。如果您在安装时遇到问题，请尝试其他不需要它的演示。

# 获取对网络摄像头 #0 的引用（默认）
video_capture = cv2.VideoCapture(0)
"""
# 加载示例图片并学习如何识别它。
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# 加载第二张示例图片并学习如何识别它。
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# 创建已知人脸编码及其名称的数组
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]
"""
save_path="./train_dir/myself/"
i=0

while i<20:
    i+=1
     # 抓取一帧视频
    ret, frame = video_capture.read()

    picture = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    picture.save(os.path.join(save_path,"{}.jpg".format(i)))

    time.sleep(2) 
    """
    # 将图像从 BGR 颜色（OpenCV 使用的）转换为 RGB 颜色（face_recognition 使用的）
    rgb_frame = frame[:, :, ::-1]

    
    # 找到视频帧中的所有人脸和人脸编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 遍历这一帧视频中的每个人脸
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 查看人脸是否与已知人脸匹配
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # 如果在 known_face_encodings 中找到匹配，则使用第一个。
        # 如果匹配中为真：
        # first_match_index =matches.index(True)
        # name = known_face_names[first_match_index]

        # 或者，使用与新人脸距离最小的已知人脸
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # 在脸部周围画一个框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 在人脸下方画一个带有名字的标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
 

    # 显示结果图像
    cv2.imshow('Video', frame)

    # 按键盘上的“q”退出！
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """


# 释放网络摄像头的句柄
video_capture.release()
cv2.destroyAllWindows()