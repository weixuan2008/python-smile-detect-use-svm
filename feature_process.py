#!/usr/bin/python 
# 这个文件的内容在公共领域。参见 LICENSE_FOR_EXAMPLE_PROGRAMS.txt 
# 
# 这个示例程序展示了如何在图像中找到正面人脸并
# 估计他们的姿势。姿势采用 68 个地标的形式。这些是
# 面部的点，例如嘴角、眉毛、眼睛等。
# 
# 我们使用的人脸检测器是使用经典的定向直方图
# 梯度 (HOG) 特征结合线性分类器、图像金字塔、
# 和滑动窗口检测方案制成的。姿势估计器是由
# # 使用 dlib 的论文实现创建的：
## 
# Vahid Kazemi 和 Josephine Sullivan，CVPR 2014 
# 与回归树集合的一毫秒人脸对齐
# # 并在 iBUG 300-W 人脸地标数据集上进行了训练（参见# https://ibug.doc.ic.ac.uk /resources/facial-point-annotations/)：   
#C. Sagonas、E. Antonakos、G、Tzimiropoulos、S. Zafeiriou、M. Pantic。
# 300 面临 In-the-wild 挑战：数据库和结果。
# Image and Vision Computing (IMAVIS)，面部地标定位“In-The-Wild”特刊。2016. 
# 你可以从以下位置获得训练好的模型文件：
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2。
# 请注意，iBUG 300-W 数据集的许可不包括商业用途。
# 所以你应该联系伦敦帝国理工学院，看看
# 
# 
# 另外，请注意，您可以使用 dlib 的机器学习

# 
# 编译/安装 DLIB PYTHON 接口
# 您可以使用以下命令安装 dlib：
# pip install dlib 
# 
# 或者，如果您想自己编译 dlib，则进入 dlib 
# 根文件夹并运行：
# python setup.py install 
# 
# 编译 dlib 应该可以在任何操作系统上运行，只要你有
# CMake 已安装。在 Ubuntu 上，这可以通过运行
# 命令
# # sudo apt-get install cmake 
# 
# 另请注意，此示例需要可以
# pip install numpy 

import sys
import os
import dlib
import glob
"""
if len(sys.argv) != 3:
    print(
        "将训练后的形状预测模型的路径作为第一个 " 
        "参数，然后是包含面部图像的目录。\n "
        "例如，如果您在 python_examples 文件夹中，则" 
        "通过运行以下命令执行此程序：\n " 
        " ./ face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces \n " 
        "您可以下载经过训练的面部形状预测器来自：\n " 
        " http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()
"""

predictor_path = "./train_dir/shape_predictor_68_face_landmarks.dat"
faces_folder_path = "./train_dir/face/"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

fi=open("./train_dir/face_feature3.txt","a")
#i=0

#for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
for i in range(4000):
    f=faces_folder_path+'file'+'{:0>4d}'.format(i+1)+".jpg"
    #print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    #win.clear_overlay()
    #win.set_image(img)

    # 让检测器找到每个人脸的边界框。
    # 第二个参数中的 1表示我们应该对图像进行 1 次上采样。这个
    # # 将使一切变得更大，并允许我们检测更多的人脸。
    dets = detector(img, 1)
    #print("Number of faces detected: {}".format(len(dets)))
    won=False
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
        # 得到的地标/用于面部在框d的部分。
        shape = predictor(img, d)
        #print('/*********************/')
        #print(shape.rect)
        rect_w=shape.rect.width()
        rect_h=shape.rect.height()
        top=shape.rect.top()
        left=shape.rect.left()
        #print(rect_w)
        #print(top)
        #print(left)
        #print('/*********************/')
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                          shape.part(1)))
        # 在屏幕上绘制人脸地标。
        #win.add_overlay(shape)
        fi.write('{},'.format(i))
        won=True
        for i in range(shape.num_parts):
            fi.write("{},{},".format((shape.part(i).x-left)/rect_w,(shape.part(i).y-top)/rect_h))
    if(won):
        fi.write("\n")
    #win.add_overlay(dets)
    #dlib.hit_enter_to_continue()

fi.close()


