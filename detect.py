#!/usr/bin/python
# -*- coding: UTF-8 -*-
from copy import deepcopy
import os
import pickle
import numpy as np
import time
import cv2, math
import cv2.aruco as aruco

#该函数将旋转向量转换为三维角度，对应偏航，俯仰，滚动角度，偏航角是最需要的
def rotationVectorToEulerAngles(rvec):
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvec, R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:  # 偏航，俯仰，滚动
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # 偏航，俯仰，滚动换成角度
    rx = x * 180.0 / 3.141592653589793
    ry = y * 180.0 / 3.141592653589793
    rz = z * 180.0 / 3.141592653589793
    return rx, ry, rz

#下面两行是相机参数中的内参矩阵和畸变系数
mtx = np.array([[629.61554535, 0.      , 333.57279485], [0.      , 631.61712266, 229.33660831], [ 0.        , 0.        , 1.        ]])
dist = np.array(([[0.03109901, -0.0100412, -0.00944869, 0.00123176, 0.31024847]]))

#此为地图中的默认标记数据，可为空，但必须有声明，识别标记时提供了手动输入标记数据功能
map = {0:[0,0,0,'office',np.array([[1,0,0],[0,1,0],[0,0,1]]),0]}

#每次启动加载标记数据
if(os.path.exists('dict_file.pkl')):
    f_read = open('dict_file.pkl', 'rb')
    map = pickle.load(f_read)
    f_read.close()

#世界参考系原点在平面地图图片上的像素位置
origin = (1164,640)

#相机相关
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
img=cv2.imread('./map.jpg')

#用到的变量
px = 0
py = 0
angle = 0

#程序主循环部分
while True:
    _img = deepcopy(img)

    # 读取摄像头画面    
    ret, frame = cap.read()
    h1, w1 = frame.shape[:2]
    #print(h1, w1)#调试用

    # 纠正畸变
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1, x:x + w1]
    frame = dst1
    # print(newcameramtx)
    #进一步处理画面
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    #如果找到id
    if ids is not None:
        angle = 0
        position = np.zeros(3, dtype=np.float32)

        #对每个标记进行姿态估计
        for i in range(len(ids)):
            _ids = ids[i]

            #没有标记的数据时手动输入
            if(_ids[0] not in map.keys()):
                print("标记库中没有这个标记，需要进行标记标定")
                _p = [float(n) for n in list(input("输入标记相对于世界坐标系原点的距离").split(' '))]
                _place = input("所处地点")
                _angle = int(input("俯视时该标记顺时针相对世界坐标系转过的角度"))
                _angle = _angle/180*3.141592653589793
                _cos = math.cos(_angle)
                _sin = math.sin(_angle)
                map[_ids[0]] = [_p[0],_p[1],_p[2],_place,np.array([[_cos,0,-_sin],[0,1,0],[_sin,0,_cos]]),_angle]
            _corners = (corners[i],)
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(_corners, 0.2, mtx, dist)
            #综合各个标记的位姿得到更准确的位置和朝向
            angle += 1/len(ids)*(rvec[0][0][2]+map[_ids[0]][5])

            # 估计每个标记的姿态并返回nt(值rvet和tvec ---不同
            # from camera coeficcients
            (rvec-tvec).any()# get rid of that nasty numpy value array error
            for i in range(rvec.shape[0]):
                cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.1)
                aruco.drawDetectedMarkers(frame, _corners)
            
            #在画面上添加一些信息
            cv2.putText(frame, "Id: " + str(_ids), (0, 40), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            EulerAngles = rotationVectorToEulerAngles(rvec)
            EulerAngles = [round(i, 2) for i in EulerAngles]
            cv2.putText(frame, "Attitude_angle:" + str(EulerAngles), (0, 120), font, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)
            tvec = tvec
            for i in range(3):
                tvec[0][0][i] = round(tvec[0][0][i], 1)
            tvec = np.squeeze(tvec)
            position += 1/len(ids)*(np.dot(map[_ids[0]][4],np.array(tvec))+np.array(map[_ids[0]][0:3]))
        cv2.putText(frame, "Position_coordinates:" + str(position) + str('m'), (0, 80), font, 0.6, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame,map[ids[0][0]][3], (0, 160), font, 0.6, (0, 255, 0), 2,
                    cv2.LINE_AA)
        
        #计算图片上的像素位置
        px=int(origin[0]-160*position[2])
        py=int(origin[1]+160*position[0])

        #平面图上画位置和朝向
        cv2.circle(_img, (px,py), 5, (0, 0, 255), -1)
        cv2.arrowedLine(_img, (px, py), (px+int(10*math.cos(angle)),py+int(10*math.sin(angle))), (0,0,255),2,0,0,0.2)
    else:
        #输出找不到标记信息，但会保留最后一次的位置在平面图上
        cv2.putText(frame, "No Ids", (0, 40), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(_img, (px,py), 5, (0, 0, 255), -1)
        cv2.arrowedLine(_img, (px, py), (px+int(10*math.cos(angle)),py+int(10*math.sin(angle))), (0,0,255),2,0,0,0.2)
    # cv2.namedWindow('frame', 0)
    # cv2.resizeWindow("frame", 960, 720)
    # 显示结果框架
    cv2.imshow("frame", frame)
    cv2.imshow("map", _img)
    key = cv2.waitKey(1)

    if key == 27:         # 按esc键退出
        print('esc break...')
        cap.release()
        f_save = open('dict_file.pkl', 'wb')
        pickle.dump(map, f_save)
        f_save.close()
        cv2.destroyAllWindows()
        break
    num = 0
    if key == ord(' '):   # 按空格键保存
        filename = "C:/Users/lcy/Pictures/" + str(time.time())[:10] + ".jpg"
        num += 1
        cv2.imwrite(filename, frame)
        print("ok")

