# 导入包
import numpy as np
import cv2
import dlib
import os
import sys
import random

def get_detector_and_predicyor():
    #使用dlib自带的frontal_face_detector作为我们的特征提取器
    detector = dlib.get_frontal_face_detector()
    """
    功能：人脸检测画框
    参数：PythonFunction和in Classes
    in classes表示采样次数，次数越多获取的人脸的次数越多，但更容易框错
    返回值是矩形的坐标，每个矩形为一个人脸（默认的人脸检测器）
    """
    #返回训练好的人脸68特征点检测器
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    return detector,predictor
#获取检测器
detector,predictor=get_detector_and_predicyor()

def painting_sunglasses(img,detector,predictor):   
    #给人脸带上墨镜
    rects = detector(img_gray, 0)  
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        right_eye_x=0
        right_eye_y=0
        left_eye_x=0
        left_eye_y=0
        for i in range(36,42):#右眼范围
            #将坐标相加
            right_eye_x+=landmarks[i][0,0]
            right_eye_y+=landmarks[i][0,1]
        #取眼睛的中点坐标
        pos_right=(int(right_eye_x/6),int(right_eye_y/6))
        """
        利用circle函数画圆
        函数原型      
        cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
        img：输入的图片data
        center：圆心位置
        radius：圆的半径
        color：圆的颜色
        thickness：圆形轮廓的粗细（如果为正）。负厚度表示要绘制实心圆。
        lineType： 圆边界的类型。
        shift：中心坐标和半径值中的小数位数。
        """
        cv2.circle(img=img, center=pos_right, radius=30, color=(0,0,0),thickness=-1)
        for i in range(42,48):#左眼范围
           #将坐标相加
            left_eye_x+=landmarks[i][0,0]
            left_eye_y+=landmarks[i][0,1]
        #取眼睛的中点坐标
        pos_left=(int(left_eye_x/6),int(left_eye_y/6))
        cv2.circle(img=img, center=pos_left, radius=30, color=(0,0,0),thickness=-1)

camera = cv2.VideoCapture(0)#打开摄像头
ok=True
# 打开摄像头 参数为输入流，可以为摄像头或视频文件
while ok:
    ok,img = camera.read()
     # 转换成灰度图像
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #display_feature_point(img,detector,predictor)
    painting_sunglasses(img,detector,predictor)#调用画墨镜函数
    cv2.imshow('video', img)
    k = cv2.waitKey(1)
    if k == 27:    # press 'ESC' to quit
        break
camera.release()
cv2.destroyAllWindows()