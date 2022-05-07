from cgi import test
import sys
import os
from PyQt5.QtGui import *
import cv2
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from detect2 import *
from search2 import *
# from search import detect
import argparse
import time
from sys import platform

from models import *
# from utils.datasets2 import *
from ui_yolov3.datasets import *
from utils.utils import *

from reid.data import make_data_loader
from reid.data.transforms import build_transforms
from reid.modeling import build_model
from reid.config import cfg as reidCfg
"""
author: lyl
date: 3/22/2022

version: torch1.8.0+cu111 1.7.0会报视图被更改的错误
"""


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.timer_camera1 = QtCore.QTimer() 
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.video = Video()

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 采用QHBoxLayout类，按照从左到右的顺序来添加控件
        self.__layout_mainmain  = QtWidgets.QVBoxLayout()
        self.__layout_personmain  = QtWidgets.QVBoxLayout()
        self.__layout_fun_button = QtWidgets.QHBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # QVBoxLayout类垂直地摆放小部件

        self.button_open_camera = QtWidgets.QPushButton(u'打开摄像头')
        # self.button_open_camera1 = QtWidgets.QPushButton(u'reid')
        self.button_close = QtWidgets.QPushButton(u'退出')
        self.upload_person = QtWidgets.QPushButton(u'上传待检索行人')
        self.person_search = QtWidgets.QPushButton(u'开始检索')
        self.lixian_video_search = QtWidgets.QPushButton(u'离线视频检索')


        # button颜色修改
        button_color = [self.button_open_camera, self.button_close, self.upload_person, self.person_search, self.lixian_video_search  ]
        for i in range(3):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                           "QPushButton:hover{color:red}"
                                           "QPushButton{background-color:rgb(78,255,255)}"
                                           "QpushButton{border:2px}"
                                           "QPushButton{border_radius:10px}"
                                           "QPushButton{padding:2px 4px}")

        self.button_open_camera.setMinimumHeight(50)
        # self.button_open_camera1.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)
        self.upload_person.setMaximumWidth(200)
        self.upload_person.setMinimumHeight(50)
        
        self.person_search.setMinimumHeight(50)
        self.lixian_video_search.setMinimumHeight(50)

        # move()方法是移动窗口在屏幕上的位置到x = 500，y = 500的位置上
        self.move(500, 500)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(100, 100)

        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)
        self.label_show_camera.setText("在线检索摄像头视频流，请打开摄像头")
        self.label_show_camera.setStyleSheet("QLabel{background:pink;}"
                                 "QLabel{color:rgb(100,100,100);font-size:15px;font-weight:bold;font-family:宋体;}"
                                 )



        self.label_show_image = QtWidgets.QLabel()
        self.label_show_image.setFixedSize(180, 250)
        self.label_show_image.setText("请上传待检索行人图片")
        self.label_show_image.setStyleSheet("QLabel{background:yellow;}"
                                 "QLabel{color:rgb(100,100,100);font-size:15px;font-weight:bold;font-family:宋体;}"
                                 )
        self.label_show_image.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        # self.__layout_fun_button.addWidget(self.button_open_camera1)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.person_search)
        self.__layout_fun_button.addWidget(self.lixian_video_search)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_personmain.addWidget(self.upload_person)


        self.__layout_mainmain.addWidget(self.label_show_image)
        self.__layout_mainmain.addLayout(self.__layout_personmain)
        self.__layout_mainmain.addLayout(self.__layout_fun_button)
        self.__layout_main.addLayout(self.__layout_mainmain)
        self.__layout_main.addWidget(self.label_show_camera)
        

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'在线行人检索系统设计')


        # # 设置背景颜色
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(),QBrush(QPixmap(r'C:\Users\Administrator\Desktop\bizhi.png')))
        self.setPalette(palette1)


    def slot_init(self):  # 建立通信连接
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        # self.button_open_camera1.clicked.connect(self.button_open_camera_click1)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_camera1.timeout.connect(self.show_camera1)
        self.button_close.clicked.connect(self.close)
        self.upload_person.clicked.connect(self.upload_person_click)
        self.person_search.clicked.connect(self.person_search_click)
        self.lixian_video_search.clicked.connect(self.lixian_video_search_click)

    def lixian_video_search_click(self):
        # app = QApplication(sys.argv)
        # my = Video()
        # my.show()
        # my.exec_()
        # sys.exit(app.exec_())
        self.anotherwindow = Video()
        self.anotherwindow.show()
        

    




    def upload_person_click(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print("imgName：",imgName) #img_pth
        jpg = QtGui.QPixmap(imgName).scaled(self.label_show_image.width(), self.label_show_image.height())
        self.label_show_image.setPixmap(jpg)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                # if msg==QtGui.QMessageBox.Cancel:
                #                     pass
            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'关闭摄像头')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'打开摄像头')

    def button_open_camera_click1(self):
        if self.timer_camera1.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                # if msg==QtGui.QMessageBox.Cancel:
                #                     pass
            else:
                self.timer_camera1.start(30)
                # self.button_open_camera1.setText(u'关闭摄像头')
        else:
            self.timer_camera1.stop()
            self.cap.release()
            self.label_show_camera.clear()
            # self.button_open_camera1.setText(u'打开摄像头')



    def show_camera(self):#行人检测
        flag, self.image = self.cap.read()  # 从视频流中读取 
        show = cv2.resize(self.image, (641, 481))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        cv2.flip(show,1,show)#水平翻转

        show = main_detect(show)

        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage


        ########################人脸检测#########################
        # faceCascade = cv2.CascadeClassifier(r"F:\biyesheji\xiaolunwen\lyl_reid\haarcascade_russian_plate_number.xml")
        # gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
        # gray = cv2.medianBlur(gray, 5)  # 中值滤波，这个中值滤波，小伙伴可以尝试将它去掉看看
        # img = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波降噪处理,是不是感觉有图像处理的地方，就有高斯滤波
        # th1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17,
        #                             6)  # 自适应二值化  # 自适应二值化处理
        # # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        # faceRects = faceCascade.detectMultiScale(th1, scaleFactor=1.1, minNeighbors=4)  # 这几个参数我调了好几次，个人感觉只能这样了

        # if len(faceRects) > 0:
        #     # 历遍每次检测的所有脸
        #     for face in faceRects:
        #         x, y, w, h = face  # face是一个元祖，返回了分类器的检测结果，包括起始点的坐标和高度宽度
        #         cv2.rectangle(show, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 3)  # 绘制人脸检测的线框（还是原谅色）
        #         font = cv2.FONT_HERSHEY_SIMPLEX  # 创建摄像头前置的文字框
        #         cv2.putText(show, 'face', (x - 30, y - 30), font, 1, (0, 255, 0), 3)#原谅色
        ########################人脸检测#########################

    def show_camera1(self):#行人检索
        flag, self.image = self.cap.read()  # 从视频流中读取

        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        cv2.flip(show,1,show)#水平翻转

        show = main_search(show)

        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

        ########################车牌检测#########################
        # platenumber = cv2.CascadeClassifier(r"F:\biyesheji\xiaolunwen\lyl_reid\haarcascade_frontalface_default.xml")
        # gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
        # gray = cv2.medianBlur(gray, 5)  # 中值滤波，这个中值滤波，小伙伴可以尝试将它去掉看看
        # img = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波降噪处理,是不是感觉有图像处理的地方，就有高斯滤波
        # th1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17,
        #                             6)  # 自适应二值化  # 自适应二值化处理
        # numberPlates = platenumber.detectMultiScale(th1, 1.1, 10)
        # for (x, y, w, h) in numberPlates:
        #     ratio = w / float(h)
        #     print(ratio)
        #     if ratio > 2.7 and ratio < 3.3:  # 这里过滤一下
        #         cv2.rectangle(show, (x, y), (x + w, y + h), (255, 0, 0), 3)
        #         cv2.putText(show, "Number Plate", (x, y - 5),
        #                     cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 3)#原谅色
        #         imgRoi = img[y:y + h, x:x + w]
        #         cv2.imshow("ROI", imgRoi)
        ########################车牌检测#########################

        



    def person_search_click(self):
        return 
        
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u'关闭', u'是否关闭！')
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


class Video(QWidget):
    
    def __init__(self):
        super(Video, self).__init__()
        self.frame = []  # 存图片
        self.detectFlag = False  # 检测flag
        self.cap = []
        self.timer_camera = QTimer()     #定义定时器
        
        # 外框
        self.resize(900, 650)
        self.setWindowTitle("离线视频行人检索系统")
        # 图片label
        self.label = QLabel(self)
        self.label.setText("请上传离线视频检索")
        self.label.setFixedSize(800, 450)  # width height
        self.label.move(50, 100)
        self.label.setStyleSheet("QLabel{background:pink;}"
                                 "QLabel{color:rgb(100,100,100);font-size:15px;font-weight:bold;font-family:宋体;}"
                                 )
        # 显示人数label
        self.label_num = QLabel(self)
        self.label_num.setText("等待检索...")
        self.label_num.setFixedSize(430, 40)  # width height
        self.label_num.move(200, 20)
        self.label_num.setStyleSheet("QLabel{background:yellow;}")
        # 开启视频按键
        self.btn = QPushButton(self)
        self.btn.setText("上传离线视频")
        self.btn.move(150, 570)
        self.btn.clicked.connect(self.slotStart)
        # 检测按键
        self.btn_detect = QPushButton(self)
        self.btn_detect.setText("检索目标行人")
        self.btn_detect.move(400, 570)
        self.btn_detect.setStyleSheet("QPushButton{background:red;}")  # 没检测红色，检测绿色
        self.btn_detect.clicked.connect(self.detection)
        # 关闭视频按钮
        self.btn_stop = QPushButton(self)
        self.btn_stop.setText("停止检索")
        self.btn_stop.move(700, 570)
        self.btn_stop.clicked.connect(self.slotStop)
 
    def slotStart(self):
        """ Slot function to start the progamme
            """
        videoName, _ = QFileDialog.getOpenFileName(self, "Open", "", "*.avi;;*.mp4;;All Files(*)")
        if videoName != "":  # “”为用户取消
            self.cap = cv2.VideoCapture(videoName)
            self.timer_camera.start(100)
            self.timer_camera.timeout.connect(self.openFrame)
 
    def slotStop(self):
        """ Slot function to stop the programme
            """
        if self.cap != []:
            self.cap.release()
            self.timer_camera.stop()   # 停止计时器
            self.label.setText("This video has been stopped.")
            self.label.setStyleSheet("QLabel{background:pink;}"
                                     "QLabel{color:rgb(100,100,100);font-size:15px;font-weight:bold;font-family:宋体;}"
                                     )
        else:
            self.label_num.setText("Push the left upper corner button to Quit.")
            Warming = QMessageBox.warning(self, "Warming", "Push the left upper corner button to Quit.",QMessageBox.Yes)
 
    def openFrame(self):
        """ Slot function to capture frame and process it
            """
        
        if(self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                if self.detectFlag == True:
                    
                    # 检测代码self.frame
                    self.label_num.setText("找到目标行人！")
 
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame.data,  width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(QPixmap.fromImage(q_image))
            else:
                self.cap.release()
                self.timer_camera.stop()   # 停止计时器
 
    def detection(self):
        self.detectFlag = not self.detectFlag # 取反
        if self.detectFlag == True:
            self.btn_detect.setStyleSheet("QPushButton{background:green;}")
        else:
            self.btn_detect.setStyleSheet("QPushButton{background:red;}")
    #        self.label_num.setText("There are 5 people.")
 


if __name__ == '__main__':
    App = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(App.exec_())

