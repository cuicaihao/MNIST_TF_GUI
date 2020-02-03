## GUI Design
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K

from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel)
from PyQt5.QtGui import (QPainter, QPen, QFont)
from PyQt5.QtCore import Qt
from PIL import ImageGrab, Image

import numpy as np

import sys
from PyQt5.QtWidgets import QApplication


Model_Path = "./models/2020.02.03_14.52.53.MnistModel.h5"

class MNIST_Window(QWidget):

    def __init__(self):
        super(MNIST_Window, self).__init__()
        self.setWindowTitle('Demo')
        self.resize(284, 330)  # resize设置宽高
        self.move(100, 100)    # move设置位置
        # self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

        self.pos_xy = []  #保存鼠标移动过的点


        best_model_filepath =  Model_Path
        self.Model =  load_model(best_model_filepath)

        # 添加一系列控件
        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 2, 280, 280)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('Result:', self)
        self.label_result_name.setGeometry(2, 290, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 290, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("Classify", self)
        self.btn_recognize.setGeometry(110, 290, 50, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.btn_clear = QPushButton("Clear", self)
        self.btn_clear.setGeometry(170, 290, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("CLose", self)
        self.btn_close.setGeometry(230, 290, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 30, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
        '''
        #中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

    def btn_recognize_on_clicked(self):
        bbox = (104, 104, 380, 380)
        im = ImageGrab.grab(bbox)    # 截屏，手写数字部分
        im = im.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素

        recognize_result = self.recognize_img(im)  # 调用识别函数
        print(recognize_result[0], flush=True)
        self.label_result.setText(str(recognize_result[0]))  # 显示识别结果
        self.update()

    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()


    def recognize_img(self, img):  # 手写体识别函数
        myimage = img.convert('L')  # 转换成灰度图
        img  = np.asarray(myimage).astype('float32')
        img = (255 - img)*1.0/255.0
        img_rows, img_cols = 28, 28
        img = img.reshape(img_rows, img_cols, 1)
        img = np.expand_dims(img, axis=0)
        # best_model_filepath =  "mnist.weights.best.2019.09.06_11.31.49.h5"
        # Model =  load_model(best_model_filepath)
        y = self.Model.predict(img)
        return np.argmax(y, axis=1)

# ————————————————
# 版权声明：本文为CSDN博主「雨寒sgg」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/u011389706/article/details/81460820


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MNIST_Window()
    mymnist.show()
    app.exec_()
 