# -*- coding: utf-8 -*-

# File: SVD_Video_Compression_Research
# Author: Jesse Redford
# Date: 4/9/2020


#-------------------------- System Setup -------------------------------------#
import os
import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QSlider,QLabel
from PyQt5 import QtCore, QtGui, QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import qdarkgraystyle
import face_recognition
import SVD_comp_lib
from SVD_comp_lib import* 
sys.path.append(r'C:\Users\Jesse\Desktop\OpenCV\openh264-1.6.0-win64msvc.dll')


print('Python Version', sys.version)
print('OpenCV Version',cv2.__version__)
print('Face recogntion Version', face_recognition.__version__)


#----------------------- UI MAIN WINDOW -------------------------------------# 
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(525, 386)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.image_label = QtWidgets.QLabel(Form)
        self.image_label.setObjectName("image_label")
        self.verticalLayout.addWidget(self.image_label)
        self.control_bt = QtWidgets.QPushButton(Form)
        self.control_bt.setObjectName("control_bt")
        self.verticalLayout.addWidget(self.control_bt)
        
        #------ Add buttons ------#
       # self.rank_bt = QtWidgets.QPushButton(Form)
       # self.rank_bt.setObjectName("rank_bt")
       # self.rank_bt.resize(100,100)
       # self.rank_bt.move(100,100)
       # self.verticalLayout.addWidget(self.rank_bt)
        
        
        
        

       
        #------- rank slider --------#
        self.rank_label = QLabel("Image Rank")
        self.verticalLayout.addWidget(self.rank_label)
        self.rank_label.setAlignment(Qt.AlignLeft)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setObjectName("rank_slider")
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(100)
        self.slider.valueChanged[int].connect(self.ValueContrast)
        self.verticalLayout.addWidget(self.slider)
     
        
        
        #------- block slider --------#
        ms,ns = get_divisors(np.zeros(shape=(480,640)))
        self.blocktypes = list(itertools.product(ms,ns))
        
        self.blocksize_label = QLabel("Block Size")
        self.verticalLayout.addWidget(self.blocksize_label)
        self.blocksize_label.setAlignment(Qt.AlignLeft)
        
        self.block_slider = QSlider(Qt.Horizontal)
        self.block_slider.setObjectName("block_slider")
        self.block_slider.setFocusPolicy(Qt.StrongFocus)
        self.block_slider.setTickPosition(QSlider.TicksBothSides)
        self.block_slider.setTickInterval(10)
        self.block_slider.setSingleStep(1)
        self.block_slider.setMinimum(0)
        self.block_slider.setMaximum(345)
        self.block_slider.setValue(78)
        self.block_slider.valueChanged[int].connect(self.block_value)
        self.verticalLayout.addWidget(self.block_slider)
        
        #--------------- Setup Layout ---------------------------#
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        #------------------------------------------------------#
        
    def ValueContrast(self):
            #print(self.slider.value())
            return(self.slider.value())
    
    def block_value(self):
        return self.blocktypes[self.block_slider.value()]
        

        
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Cam view"))
        self.image_label.setText(_translate("Form", "TextLabel"))
        self.control_bt.setText(_translate("Form", "Start"))
        
        self.rank_label.setText(_translate("Form", "Image Rank"))
        self.blocksize_label.setText(_translate("Form", "Block Size"))
        
        
       
        
#-----------------------------------------------------------------#     

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        self.font      =  cv2.FONT_HERSHEY_SIMPLEX            # text font for results
        self.color     =  (0, 0, 255)                         # font color (0,0,255) = black
        self.org       =  (10,25)                             # location of image metrics
        self.org_fig   =  (10,50)                             # location of figure metrics
        self.fontScale =   .35                                 # size of text 
        self.thickness =    1                                 # thickness of text, int type
        
        
        
        
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
       
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)

    # view camera
    def viewCam(self):
        
        ret, image = self.cap.read() # read image in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gs_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          
        face_locations = face_recognition.face_locations(gs_frame)
        
        #-------- process image frame ----------#

        self.k = self.ui.ValueContrast()
        self.ui.rank_label.setText('Image Rank:'+''+str(self.k))
        
        self.block = self.ui.block_value()
        self.ui.blocksize_label.setText('Block Size:'+''+str(self.block))
        
        #svd_frame  = np.uint8(k_svd(gs_frame,self.k))
        #svd_frame  = np.uint8(revised_block_svd(gs_frame,m = self.block[0], n = self.block[1]))
        
        if len(face_locations) > 0:
            svd_frame,figure_metrics,ground_metrics = Fc_SVD(gs_frame,image.shape[0],image.shape[1],self.block[0],self.block[1],face_locations)
            svd_frame = cv2.putText(svd_frame,str(figure_metrics), self.org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA, False)
            svd_frame = cv2.putText(svd_frame,str(ground_metrics), self.org_fig, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA, False)
            for x1, y1, x2, y2 in face_locations:
                cv2.rectangle(svd_frame, (y1, x1), (y1+(y2-y1), x1+(x2-x1)), (0, 255, 0), 2)
            self.comp_video.write(svd_frame)            
        else:
            B_CF = round(compression_ratio(image.shape[0],image.shape[1],self.block[0],self.block[1]),2)
            start = time.time()
            svd_frame  = np.uint8(block_svd(gs_frame,m = self.block[0], n = self.block[1]))
            end = time.time()
            TC = round(end-start,2)
            #svd_frame  = np.uint8(k_svd(gs_frame,self.k))
            
            self.comp_video.write(svd_frame)
   
           
            B_RMSE,B_PSNR = error_metrics(gs_frame,svd_frame)
            b_svd_text = 'Block Rank:' + str(1) + ' | '                  \
                + 'CF:' + str(B_CF) + ' | '                   \
                + 'TC:' + str(TC) + ' | '               \
                + 'RMSE:' + str(round(B_RMSE,2)) + ' | '       \
                + 'PSNR:' + str(round(B_PSNR,2)) + ' | '        \
                + 'Blocksize:' + str(self.block[0]) +'x'+str(self.block[0]) + ' |'
            svd_frame = cv2.putText(svd_frame,str(b_svd_text), self.org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA, False)
        
        image = svd_frame
   
 
        qImg = qImg = QImage(image.data,image.shape[1], image.shape[0], image.shape[1],  QImage.Format.Format_Indexed8)
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
        self.gs_video.write(gs_frame)

        
        
        
        #-----------orignial ----------------#
         # read image in BGR format
        #ret, image = self.cap.read()
        # convert image to RGB format
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
       # height, width, channel = image.shape
       # step = channel * width
        # create QImage from image
       # qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
       # self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
        #------------------------------------------------------#
    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop Recording")
        # if timer is started
            self.gs_video = cv2.VideoWriter('gs_stream.mp4v',cv2.VideoWriter_fourcc('H','2','6','5'), 10, (640,480),0)
            self.comp_video = cv2.VideoWriter('comp_stream.mp4v',cv2.VideoWriter_fourcc('H','2','6','5'), 10, (640,480),0)
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.control_bt.setText("Start Recording")
            
            self.gs_video.release()
            self.comp_video.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())