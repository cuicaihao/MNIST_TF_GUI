# -*- coding: utf-8 -*-

"""
Created on Friday 2 Feb 2020

@author: Chris.Cui

Email: Chris.Cui@aurecongroup.com

"""
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
from matplotlib import pyplot as plt



Model_Path = r"./models/mnist.weights.best.h5"

class MNIST_Window(QWidget):

    def __init__(self):
        super(MNIST_Window, self).__init__()
        self.setWindowTitle('MNIST')
        self.resize(284, 330)  # resize设置宽高
        self.move(100, 100)    # move设置位置
        # self.setWindowFlags(Qt.FramelessWindowHint)  # no frames
        # setMouseTracking is False, only track when pressed 
        self.setMouseTracking(False)
        self.pos_xy = []  # save the points
        best_model_filepath =  Model_Path
        self.Model =  load_model(best_model_filepath)

        # add wegites
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
        pen = QPen(Qt.black, 23, Qt.SolidLine) # line width is 25 2~3 px based on the MNIST
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
        Pressed moving add points into xy list
        '''
        # extrack the positions  
        pos_tmp = (event.pos().x(), event.pos().y())
        # pos_tmp add to self.pos_xy list
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            Overwrite the mouserelease event
            pos_x (-1, -1) add breakpoints each time
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

    def btn_recognize_on_clicked(self): # issues within screen capture.
        # bbox = (104, 104, 380, 380) / [4, 40] [284, 320]
        bbox = (114, 150, 114+280-11, 150+280-11) # relocated [100, 100] with padding 10
        im = []
        im = ImageGrab.grab(bbox)    # screen capture
        im = im.resize((28, 28), Image.ANTIALIAS)  # reshape image

        recognize_result = self.recognize_img(im)  # apply our pretrained model
        print(recognize_result[0], flush=True)
        self.label_result.setText(str(recognize_result[0]))  # show label
        self.update()

    def btn_clear_on_clicked(self): # clear the canvas
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()

    def recognize_img(self, img):   #  
        myimage = img.convert('L')  # convert the imge to grayscale format
        img  = np.asarray(myimage).astype('float32')
        img = (255 - img)*1.0/255.0
        img_rows, img_cols = 28, 28
        img = img.reshape(img_rows, img_cols, 1) # change the input format
        img = np.expand_dims(img, axis=0)
        ## review the input images 
        plt.figure()
        plt.imshow(np.squeeze(img)) # remove extra dimensions
        plt.colorbar()
        plt.grid(False)
        plt.show()
        y = self.Model.predict(img)
        return np.argmax(y, axis=1)
 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MNIST_Window()
    mymnist.show()
    app.exec_()
 