# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1079, 719)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/app_icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(190, 60, 551, 560))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setObjectName("frame")
        self.video_feed = QtWidgets.QLabel(self.frame)
        self.video_feed.setGeometry(QtCore.QRect(0, 0, 551, 560))
        self.video_feed.setText("")
        self.video_feed.setObjectName("video_feed")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 60, 171, 560))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_2.setObjectName("frame_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 370, 150, 131))
        self.groupBox_3.setObjectName("groupBox_3")
        self.smile_rect_radio = QtWidgets.QRadioButton(self.groupBox_3)
        self.smile_rect_radio.setGeometry(QtCore.QRect(20, 90, 120, 30))
        self.smile_rect_radio.setObjectName("smile_rect_radio")
        self.rect_radio_group = QtWidgets.QButtonGroup(MainWindow)
        self.rect_radio_group.setObjectName("rect_radio_group")
        self.rect_radio_group.addButton(self.smile_rect_radio)
        self.eye_rect_radio = QtWidgets.QRadioButton(self.groupBox_3)
        self.eye_rect_radio.setGeometry(QtCore.QRect(20, 60, 120, 30))
        self.eye_rect_radio.setObjectName("eye_rect_radio")
        self.rect_radio_group.addButton(self.eye_rect_radio)
        self.face_rect_radio = QtWidgets.QRadioButton(self.groupBox_3)
        self.face_rect_radio.setGeometry(QtCore.QRect(20, 30, 120, 30))
        self.face_rect_radio.setChecked(True)
        self.face_rect_radio.setObjectName("face_rect_radio")
        self.rect_radio_group.addButton(self.face_rect_radio)
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 150, 191))
        self.groupBox_2.setObjectName("groupBox_2")
        self.generate_dataset_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.generate_dataset_btn.setGeometry(QtCore.QRect(10, 30, 120, 29))
        self.generate_dataset_btn.setStyleSheet("background-color:rgb(0, 153, 229)")
        self.generate_dataset_btn.setObjectName("generate_dataset_btn")
        self.train_model_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.train_model_btn.setGeometry(QtCore.QRect(10, 70, 120, 29))
        self.train_model_btn.setStyleSheet("background-color:rgb(0, 153, 229)")
        self.train_model_btn.setObjectName("train_model_btn")
        self.recognize_face_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.recognize_face_btn.setGeometry(QtCore.QRect(10, 110, 120, 29))
        self.recognize_face_btn.setStyleSheet("background-color:rgb(0, 153, 229)")
        self.recognize_face_btn.setObjectName("recognize_face_btn")
        self.save_image_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.save_image_btn.setGeometry(QtCore.QRect(10, 150, 120, 29))
        self.save_image_btn.setStyleSheet("background-color:rgb(0, 153, 229)")
        self.save_image_btn.setObjectName("save_image_btn")
        self.groupBox_1 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_1.setGeometry(QtCore.QRect(10, 220, 150, 131))
        self.groupBox_1.setFlat(False)
        self.groupBox_1.setCheckable(False)
        self.groupBox_1.setObjectName("groupBox_1")
        self.lbph_algo_radio = QtWidgets.QRadioButton(self.groupBox_1)
        self.lbph_algo_radio.setGeometry(QtCore.QRect(20, 90, 120, 22))
        self.lbph_algo_radio.setChecked(True)
        self.lbph_algo_radio.setObjectName("lbph_algo_radio")
        self.algo_radio_group = QtWidgets.QButtonGroup(MainWindow)
        self.algo_radio_group.setObjectName("algo_radio_group")
        self.algo_radio_group.addButton(self.lbph_algo_radio)
        self.fisher_algo_radio = QtWidgets.QRadioButton(self.groupBox_1)
        self.fisher_algo_radio.setGeometry(QtCore.QRect(20, 60, 120, 22))
        self.fisher_algo_radio.setObjectName("fisher_algo_radio")
        self.algo_radio_group.addButton(self.fisher_algo_radio)
        self.eigen_algo_radio = QtWidgets.QRadioButton(self.groupBox_1)
        self.eigen_algo_radio.setGeometry(QtCore.QRect(20, 30, 120, 22))
        self.eigen_algo_radio.setObjectName("eigen_algo_radio")
        self.algo_radio_group.addButton(self.eigen_algo_radio)
        self.video_recording_btn = QtWidgets.QPushButton(self.frame_2)
        self.video_recording_btn.setGeometry(QtCore.QRect(35, 510, 87, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.video_recording_btn.setFont(font)
        self.video_recording_btn.setStyleSheet("background-color:rgb(0, 153, 229);\n"
"color:rgb(193, 0, 0)")
        self.video_recording_btn.setCheckable(True)
        self.video_recording_btn.setObjectName("video_recording_btn")
        self.frame1 = QtWidgets.QFrame(self.centralwidget)
        self.frame1.setGeometry(QtCore.QRect(10, 630, 731, 61))
        self.frame1.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame1.setObjectName("frame1")
        self.progress_bar_recognize = QtWidgets.QProgressBar(self.frame1)
        self.progress_bar_recognize.setGeometry(QtCore.QRect(490, 20, 118, 23))
        self.progress_bar_recognize.setProperty("value", 0)
        self.progress_bar_recognize.setObjectName("progress_bar_recognize")
        self.label_3 = QtWidgets.QLabel(self.frame1)
        self.label_3.setGeometry(QtCore.QRect(420, 20, 71, 21))
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(self.frame1)
        self.label.setGeometry(QtCore.QRect(230, 20, 54, 21))
        self.label.setObjectName("label")
        self.progress_bar_train = QtWidgets.QProgressBar(self.frame1)
        self.progress_bar_train.setGeometry(QtCore.QRect(290, 20, 118, 23))
        self.progress_bar_train.setMaximum(3)
        self.progress_bar_train.setProperty("value", 0)
        self.progress_bar_train.setObjectName("progress_bar_train")
        self.progress_bar_generate = QtWidgets.QProgressBar(self.frame1)
        self.progress_bar_generate.setGeometry(QtCore.QRect(100, 20, 118, 23))
        self.progress_bar_generate.setProperty("value", 0)
        self.progress_bar_generate.setObjectName("progress_bar_generate")
        self.label_2 = QtWidgets.QLabel(self.frame1)
        self.label_2.setGeometry(QtCore.QRect(30, 20, 61, 21))
        self.label_2.setObjectName("label_2")
        self.exit_btn = QtWidgets.QPushButton(self.frame1)
        self.exit_btn.setGeometry(QtCore.QRect(630, 10, 87, 31))
        self.exit_btn.setObjectName("exit_btn")
        self.label_4 = QtWidgets.QLabel(self.frame1)
        self.label_4.setGeometry(QtCore.QRect(870, 10, 54, 41))
        self.label_4.setInputMethodHints(QtCore.Qt.ImhNone)
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("icon/github.png"))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(860, 10, 181, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.header = QtWidgets.QLabel(self.centralwidget)
        self.header.setGeometry(QtCore.QRect(10, 0, 1051, 51))
        self.header.setStyleSheet("background-color:rgb(232, 155, 0)")
        self.header.setFrameShape(QtWidgets.QFrame.Panel)
        self.header.setText("")
        self.header.setObjectName("header")
        self.lbl_head2 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_head2.setGeometry(QtCore.QRect(420, 20, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.lbl_head2.setFont(font)
        self.lbl_head2.setObjectName("lbl_head2")
        self.lbl_head1 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_head1.setGeometry(QtCore.QRect(390, 0, 371, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.lbl_head1.setFont(font)
        self.lbl_head1.setObjectName("lbl_head1")
        self.logo_itenas = QtWidgets.QLabel(self.centralwidget)
        self.logo_itenas.setEnabled(True)
        self.logo_itenas.setGeometry(QtCore.QRect(10, 0, 111, 51))
        self.logo_itenas.setMinimumSize(QtCore.QSize(0, 0))
        self.logo_itenas.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.logo_itenas.setStyleSheet("image: url(:/newPrefix/itenas.png);\n"
"image: url(:/newPrefix/itenas.png);")
        self.logo_itenas.setText("")
        self.logo_itenas.setPixmap(QtGui.QPixmap(":/newPrefix/itenas.png"))
        self.logo_itenas.setObjectName("logo_itenas")
        self.header.raise_()
        self.frame.raise_()
        self.frame_2.raise_()
        self.frame.raise_()
        self.label_5.raise_()
        self.lbl_head2.raise_()
        self.lbl_head1.raise_()
        self.logo_itenas.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menu_bar = QtWidgets.QMenuBar(MainWindow)
        self.menu_bar.setGeometry(QtCore.QRect(0, 0, 1079, 21))
        self.menu_bar.setDefaultUp(False)
        self.menu_bar.setObjectName("menu_bar")
        MainWindow.setMenuBar(self.menu_bar)

        self.retranslateUi(MainWindow)
        self.exit_btn.clicked['bool'].connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pengenalan Micro Expression Wajah Menggunakan Local Binary Pattern"))
        self.groupBox_3.setTitle(_translate("MainWindow", "RECTANGLE"))
        self.smile_rect_radio.setText(_translate("MainWindow", "Smile"))
        self.eye_rect_radio.setText(_translate("MainWindow", "Eye"))
        self.face_rect_radio.setText(_translate("MainWindow", "Face"))
        self.groupBox_2.setTitle(_translate("MainWindow", "ACTIONS"))
        self.generate_dataset_btn.setText(_translate("MainWindow", "Generate Dataset"))
        self.train_model_btn.setText(_translate("MainWindow", "Train Model"))
        self.recognize_face_btn.setText(_translate("MainWindow", "Recognize Face"))
        self.save_image_btn.setText(_translate("MainWindow", "Save Image"))
        self.groupBox_1.setTitle(_translate("MainWindow", "ALGORITHMS"))
        self.lbph_algo_radio.setText(_translate("MainWindow", "LBPH"))
        self.fisher_algo_radio.setText(_translate("MainWindow", "Fisherfaces"))
        self.eigen_algo_radio.setText(_translate("MainWindow", "Eignefaces"))
        self.video_recording_btn.setText(_translate("MainWindow", "Record"))
        self.label_3.setText(_translate("MainWindow", "Confidence"))
        self.label.setText(_translate("MainWindow", "Trainined"))
        self.progress_bar_train.setFormat(_translate("MainWindow", "%p%"))
        self.label_2.setText(_translate("MainWindow", "Generated"))
        self.exit_btn.setText(_translate("MainWindow", "Exit"))
        self.label_5.setText(_translate("MainWindow", "Gifan Arief Caesar - 152015062"))
        self.lbl_head2.setText(_translate("MainWindow", "Menggunakan Local Binary Pattern"))
        self.lbl_head1.setText(_translate("MainWindow", "Aplikasi Pengenalan Micro Expression Wajah"))
