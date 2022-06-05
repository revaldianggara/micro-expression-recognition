import os
from datetime import datetime
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox, QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import math as m
from PIL import Image
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt

from PIL import ImageTk, Image
import face_recognition as fr

from PyQt5 import QtGui
import sys
 
from PyQt5.QtGui import QPixmap

font = cv2.FONT_HERSHEY_SIMPLEX
facedetect = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# class USER(QDialog):        # Dialog box for entering name and key of new dataset.
#     """USER Dialog """
#     def __init__(self):
#         super(USER, self).__init__()
#         loadUi("user_info.ui", self)
#
#     def get_name_key(self):
#         # name = self.name_label.text()
#         key = int(self.key_label.text())
#         return name, key
class Visualize_LBP():
    def __init__(self,Radius=1,neighbors=4):
        self._radius = Radius
        self._neighbors = neighbors
    @property
    def radius(self):
        return self._radius
    @property
    def neighbors(self):
        return self._neighbors
    def compute_LBP(self,img):
        img1 = img.copy()
        for i in range(m.ceil(self._radius),img.shape[0] - m.ceil(self._radius)):
            for j in range(m.ceil(self._radius),img.shape[1] - m.ceil(self._radius)):
                window = img[i - m.ceil(self._radius) : i + m.ceil(self._radius) + 1,
                j - m.ceil(self._radius) : j + m.ceil(self._radius) + 1]
                window_lbp =  self.LBP(window)
                img[i,j] = window_lbp
                cv2.imshow('faces', img)
                #cv2.imshow('faces', img1)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    sys.exit(0)
    def LBP(self,img_chunk):
        #coordinates, using ceil because the block is bigger than circle
        gc_x = m.ceil(self._radius)
        gc_y = gc_x
        #pixel value
        gc = img_chunk[gc_x, gc_y]
        print("gc=====",gc)
        gp_xs = []
        gp_ys = []
        pixels = []
        angles = []
        lbp_result = 0
        for p in range(self._neighbors):
            theta = 2 * m.pi * p /self._neighbors
            angles.append(theta)
            gp_x = - self._radius * m.sin(theta)
            gp_x = round(gp_x,4)#get 4 decimals
            gp_xs.append(gp_x)
            gp_y = + self._radius * m.cos(theta)
            gp_y = round(gp_y,4)#get 4 decimals
            gp_ys.append(gp_y)
            #get the fraction part of gp_x
            gp_x_fract = gp_x - m.floor(gp_x)
            #get the fraction part of gp_y
            gp_y_fract = gp_y - m.floor(gp_y)
            gp_x_trans = round(gp_x + m.ceil(self._radius),4)
            gp_y_trans = round(gp_y + m.ceil(self._radius),4)
            #if the fraction parts are zeros, then no need to interpolate
            if gp_x_fract == 0 and gp_y_fract == 0:
                pixel = img_chunk[int(gp_x_trans),int(gp_y_trans)]
                pixels.append(pixel)
            else:
                points = []
                if 0 <= theta <=  m.pi/2:
                    q11 = gc
                    points.append(q11)
                    q21 = img_chunk[gc_y , -1 ]
                    points.append(q21)
                    q22 = img_chunk[0,gc_x * 2 ]
                    points.append(q22)
                    q12 = img_chunk[0,gc_x]
                    points.append(q12)
                    x1 = y1 = 0
                    x2 = y2 = gc_x
                    points.append(x1)
                    points.append(y1)
                    points.append(x2)
                    points.append(y2)
                    #[q11,q21,q22,q12,x1,y1,x2,y2]
                    pixel = int(self.bilinear_interpolation(abs(gp_y),abs(gp_x),points))
                if m.pi/2 < theta <=  m.pi:
                    q11 = img_chunk[gc_x , 0 ]
                    points.append(q11)
                    q21 = gc
                    points.append(q21)
                    q22 = img_chunk[0,gc_x]
                    points.append(q22)
                    q12 = img_chunk[0,0]
                    points.append(q12)
                    x1 = y1 = 0
                    x2 = y2 = gc_x
                    points.append(x1)
                    points.append(y1)
                    points.append(x2)
                    points.append(y2)
                    #[q11,q21,q22,q12,x1,y1,x2,y2]
                    pixel = int(self.bilinear_interpolation(abs(gp_x_trans),abs(gp_y_trans),points))
                if m.pi < theta <=  3*m.pi/2:
                    q11 = img_chunk[-1 , 0]
                    points.append(q11)
                    q21 = img_chunk[-1 , gc_x]
                    points.append(q21)
                    q22 = gc
                    points.append(q22)
                    q12 = img_chunk[gc_x,0]
                    points.append(q12)
                    x1 = y1 = 0
                    x2 = y2 = gc_x
                    points.append(x1)
                    points.append(y1)
                    points.append(x2)
                    points.append(y2)
                    #[q11,q21,q22,q12,x1,y1,x2,y2]
                    pixel = int(self.bilinear_interpolation(abs(gp_x_trans),abs(gp_x_trans),points))
                if 3*m.pi/2 < theta <=  2*m.pi:
                    q11 = img_chunk[-1 , gc_x]
                    points.append(q11)
                    q21 = img_chunk[-1 , -1]
                    points.append(q21)
                    q22 = img_chunk[gc_x , -1]
                    points.append(q22)
                    q12 = gc
                    points.append(q12)
                    x1 = y1 = 0
                    x2 = y2 = gc_x
                    points.append(x1)
                    points.append(y1)
                    points.append(x2)
                    points.append(y2)
                    #[q11,q21,q22,q12,x1,y1,x2,y2]
                    pixel = int(self.bilinear_interpolation(gp_x,gp_y,points))
            lbp_result += self.pixel_compare(gc,pixel)*2**p
        return lbp_result
    def bilinear_interpolation(self,x,y,points):
        q11,q21,q22,q12,x1,y1,x2,y2 = points
        return (q11 * (x2 - x) * (y2 - y) + q21 * (x - x1) * (y2 - y) + q12 * (x2 - x) * (y - y1) + q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
    def pixel_compare(self,gc,pixel):
        return 1 if pixel >= gc else 0

class Fast_LBP():
    def __init__(self, radius=1, neighbors=8):
        self._radius = radius
        self._neighbors = neighbors
    @property
    def radius(self):
        return self._radius
    @property
    def neighbors(self):
        return self._neighbors
    def Compute_LBP(self,Image):
        #Determine the dimensions of the input image.
        ysize, xsize = Image.shape
        # define circle of symetrical neighbor points
        angles_array = 2*np.pi/self._neighbors
        alpha = np.arange(0,2*np.pi,angles_array)
        # Determine the sample points on circle with radius R
        s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
        s_points *= self._radius
        # s_points is a 2d array with 2 columns (y,x) coordinates for each cicle neighbor point
        # Determine the boundaries of s_points wich gives us 2 points of coordinates
        # gp1(min_x,min_y) and gp2(max_x,max_y), the coordinate of the outer block
        # that contains the circle points
        min_y=min(s_points[:,0])
        max_y=max(s_points[:,0])
        min_x=min(s_points[:,1])
        max_x=max(s_points[:,1])
        # Block size, each LBP code is computed within a block of size bsizey*bsizex
        # so if radius = 1 then block size equal to 3*3
        bsizey = np.ceil(max(max_y,0)) - np.floor(min(min_y,0)) + 1
        bsizex = np.ceil(max(max_x,0)) - np.floor(min(min_x,0)) + 1
        # Coordinates of origin (0,0) in the block
        origy =  int(0 - np.floor(min(min_y,0)))
        origx =  int(0 - np.floor(min(min_x,0)))
        #Minimum allowed size for the input image depends on the radius of the used LBP operator.
        if xsize < bsizex or ysize < bsizey :
            raise Exception('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')
        # Calculate dx and dy: output image size
        # for exemple, if block size is 3*3 then we need to substract the first row and the last row which is 2 rows
        # so we need to substract 2, same analogy applied to columns
        dx = int(xsize - bsizex + 1)
        dy = int(ysize - bsizey + 1)
        # Fill the center pixel matrix C.
        C = Image[origy:origy+dy,origx:origx+dx]
        # Initialize the result matrix with zeros.
        result = np.zeros((dy,dx), dtype=np.float32)
        for i in range(s_points.shape[0]):
            # Get coordinate in the block:
            p = s_points[i][:]
            y,x = p + (origy, origx)
            # Calculate floors, ceils and rounds for the x and ysize
            fx = int(np.floor(x))
            fy = int(np.floor(y))
            cx = int(np.ceil(x))
            cy = int(np.ceil(y))
            rx = int(np.round(x))
            ry = int(np.round(y))
            D = [[]]
            if np.abs(x - rx) < 1e-6 and np.abs(y - ry) < 1e-6:
            #Interpolation is not needed, use original datatypes
                N = Image[ry:ry+dy,rx:rx+dx]
                D = (N >= C).astype(np.uint8)
            else:
                # interpolation is needed
                # compute the fractional part.
                ty = y - fy
                tx = x - fx
                # compute the interpolation weight.
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty
                # compute interpolated image:
                N = w1*Image[fy:fy+dy,fx:fx+dx]
                N = np.add(N, w2*Image[fy:fy+dy,cx:cx+dx], casting="unsafe")
                N = np.add(N, w3*Image[cy:cy+dy,fx:fx+dx], casting="unsafe")
                N = np.add(N, w4*Image[cy:cy+dy,cx:cx+dx], casting="unsafe")
                D = (N >= C).astype(np.uint8)
            #Update the result matrix.
            v = 2**i
            result += D*v
        return result.astype(np.uint8)


class FacialExpression(QMainWindow, QWidget):        # Main application
    """Main Class"""
    def __init__(self, parent=None):
        super(FacialExpression, self).__init__(parent)
        loadUi("mainwindow.ui", self)
        # Classifiers, frontal face, eyes and smiles.
        self.face_classifier = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
        self.eye_classifier = cv2.CascadeClassifier("classifiers/haarcascade_eye.xml")
        self.smile_classifier = cv2.CascadeClassifier("classifiers/haarcascade_smile.xml")

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = self.figure.add_subplot(111)

        self.layoutvertical = QVBoxLayout(self)
        self.layoutvertical.addWidget(self.canvas)

        # Variables
        self.camera_id = 0 # can also be a url of Video
        self.dataset_per_subject = 1
        self.ret = False
        self.trained_model = 0

        self.image = cv2.imread("icon/icon_diagram.png", 1)
        self.modified_image = self.image.copy()
        self.draw_text("Authenticate Using Face Recognition", 40, 30, 1, (255,255,255))
        self.display()
        # Actions
        self.generate_dataset_btn.setCheckable(True)
        self.train_model_btn.setCheckable(True)
        self.recognize_face_btn.setCheckable(True)
        # Menu
        self.about_menu = self.menu_bar.addAction("About")
        self.help_menu = self.menu_bar.addAction("Help")
        self.about_menu.triggered.connect(self.about_info)
        self.help_menu.triggered.connect(self.help_info)
        # Algorithms
        self.algo_radio_group.buttonClicked.connect(self.algorithm_radio_changed)
        # Recangle
        self.face_rect_radio.setChecked(True)
        self.eye_rect_radio.setChecked(False)
        self.smile_rect_radio.setChecked(False)
        self.lbph_algo_radio.setVisible(False)
        self.label.setVisible(False)
        self.progress_bar_train.setVisible(False)
        self.label_3.setVisible(False)
        self.progress_bar_recognize.setVisible(False)
        # Events
        self.generate_dataset_btn.clicked.connect(self.generate)
        self.train_model_btn.clicked.connect(self.train)
        self.recognize_face_btn.clicked.connect(self.recognize)
        self.histogram_btn.clicked.connect(self.lbph_show)
        self.cat.clicked.connect(self.show_cat)
        self.detect_btn.clicked.connect(self.detect)
        self.detect_btn_2.clicked.connect(self.detect_2)
        # self.detect_btn.clicked.connect(self.compareHist)
        # Recognizers
        self.update_recognizer()
        self.assign_algorithms()

        self.photo.setPixmap(QtGui.QPixmap("icon/noimage.png"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        self.b_img.clicked.connect(self.getImage)

        self.cat.setObjectName("cat")
        self.train_model_btn.setVisible(False)
        
    def getImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','c:\\', "Image files (*.jpg *.jpeg *.png)")
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)
        # self.label.setPixmap(QPixmap(pixmap))
        self.photo.setPixmap(QtGui.QPixmap(pixmap))
        img = cv2.imread(imagePath)
        cv2.imwrite("pilih_gambar/selected.jpg", img)

    def show_cat(self):
        pixel_value = []
        self.photo.setPixmap(QtGui.QPixmap("capture/1.jpg"))
        img = cv2.imread(r'capture/1.jpg')
        img = print(img)
        img = pixel_value.append(img)
        self.frame_pixel.setText(img)

    def detect(self, id):
        test_img=cv2.imread(r'capture/1.jpg')      #Give path to the image which you want to test
        face_detected = None
        faces_detected,gray_img=fr.faceDetection(test_img)
        print("face Detected: ",faces_detected)

        if face_detected == None:
            self.lbl_hasil.setText("Ekspresi tidak dikenali \n atau Gambar tidak jelas")
            self.lbl_hasil_confidence.setText("..........")

        face_recognizer=cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(r'training/lbph_trained_model.yml')  #Give path of where trainingData.yml is saved

        name={0:"Anger",1:"Contemp", 2:"Disgust", 3:"Fear", 4:"Happy", 5:"Sadness", 6:"Surprise"}              #Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.

        for face in faces_detected:
            (x,y,w,h)=face
            roi_gray=gray_img[y:y+h,x:x+h]
            label,euclid=face_recognizer.predict(roi_gray)
            # euc = distance.euclidean(hist1, hist2)
            # matrix1, matrix2, disparity = procrustes(hist1, hist2)
            self.lbl_hasil_confidence.setText(str(f"{euclid / 1000 :.2f}"))
            print("label :",label)
            fr.draw_rect(test_img,face)
            predicted_name = label
            # fr.put_text(test_img,predicted_name,x,y)
            self.lbl_hasil.setText(str(name[predicted_name]))

        resized_img=cv2.resize(test_img,(500,500))
        cv2.imwrite("hasil_deteksi/detected.jpg", resized_img)
        self.photo.setPixmap(QtGui.QPixmap("hasil_deteksi/detected.jpg"))

    # def euclidean(self, vector):
    #     d = 0
    #     v = vector.reshape(1, -1)
    #     print(v, 'vector matrix')
    #     for s, vc in zip(self, vector):
    #         d += (s - vc) * (s - vc)
    #     return m.sqrt(d)

    def detect_2(self, id):
        test_img=cv2.imread(r'pilih_gambar/selected.jpg')      #Give path to the image which you want to test

        face_detected2 = None
        faces_detected2,gray_img=fr.faceDetection(test_img)
        print("face Detected: ",faces_detected2)

        if face_detected2 == None:
            self.lbl_hasil.setText("Ekspresi tidak dikenali \n atau Gambar tidak jelas")
            self.lbl_hasil_confidence.setText("..........")

        face_recognizer=cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(r'training/lbph_trained_model.yml')  #Give path of where trainingData.yml is saved

        name2={0:"Anger",1:"Contemp", 2:"Disgust", 3:"Fear", 4:"Happy", 5:"Sadness", 6:"Surprise"}              #Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.

        # kelas2 = {}
        # for i in name:
        #     kelas2[name[i]] = i

        for face in faces_detected2:
            (x,y,w,h)=face
            roi_gray=gray_img[y:y+h,x:x+h]
            label2,euclid=face_recognizer.predict(roi_gray)
            # euc = distance.euclidean(hist1, hist2)
            # matrix1, matrix2, disparity = procrustes(hist1, hist2)
            # print ("Confidence :",confidence)
            self.lbl_hasil_confidence.setText(str(f"{euclid / 1000 :.2f}"))
            print("label :",label2)
            fr.draw_rect(test_img,face)
            predicted_name2 = label2
            # predicted_name=name[label2]
            # fr.put_text(test_img,predicted_name,x,y)
            self.lbl_hasil.setText(str(name2[predicted_name2]))

        resized_img=cv2.resize(test_img,(500,500))
        cv2.imwrite("hasil_deteksi/detected.jpg", resized_img)
        self.photo.setPixmap(QtGui.QPixmap("hasil_deteksi/detected.jpg"))

    #Menampilkan proses LBP
    def lbph_show(self):
        class LBP_sklearn:
            def __init__(self, Nb_Points, Radius):
                # Initiate number of points(neighbors) and the radius of the cercle
                self._Nb_Points = Nb_Points
                self._Radius = Radius

            @property
            def Radius(self):
                return self._Radius

            @property
            def Nb_Points(self):
                return self._Nb_Points

            def compute(self, gray):
                # compute the Local Binary Pattern of the image,
                # and then use the LBP representation
                # to build the histogram of patterns
                LBP = local_binary_pattern(gray, self._Nb_Points, self._Radius, method="uniform")
                axs[1][1].imshow(LBP, cmap='gray', vmin=0, vmax=9)
                axs[1][1].set_title('LBP Image', fontdict={'fontsize': 15, 'fontweight': 'medium'})
                axs[1][1].axis('off')
                (hist, bins) = np.histogram(LBP.ravel(),
                                            bins=np.arange(0, self._Nb_Points + 3),
                                            range=(0, self._Nb_Points + 2))
                width = bins[1] - bins[0]
                center = (bins[:-1] + bins[1:]) / 2
                axs[1][0].bar(center, hist, align='center', width=width)
                axs[1][0].set_title('Histogram', fontdict={'fontsize': 15, 'fontweight': 'medium'})
                # normalize the histogram
                hist = hist.astype("float")
                hist /= hist.sum()
                # return the histogram of Local Binary Patterns
                return hist
        var_lbp = Visualize_LBP(Radius=1, neighbors=8)
        matrix1 = np.array([[25, 41, 24], [29, 33, 80], [38, 56, 65]], dtype=np.int16)
        matrix2 = np.array(
            [[50, 60, 70, 80, 90], [51, 61, 2, 81, 3], [52, 62, 72, 82, 92], [53, 13, 73, 43, 93], [54, 64, 74, 22, 1]],
            dtype=np.uint8)
        # print(matrix2)
        lbp2 = Fast_LBP(1, 8)
        # result = lbp2.Compute_LBP(matrix2)
        # print("result2 = ",result)
        # var_lbp.compute_LBP(matrix2)

        # print("result2 = \n",result)

        img = np.array(cv2.imread("hasil_deteksi/detected.jpg", cv2.IMREAD_GRAYSCALE), 'uint8')
        img1 = np.array(cv2.imread("hasil_deteksi/detected.jpg"), 'uint8')
        cv2.imshow('faces', img)
        if cv2.waitKey(1000) & 0xFF == ord("q"):
            pass
        # result = var_lbp.compute_LBP(img)
        result1 = lbp2.Compute_LBP(img)
        # cv2.imshow('faces', result)
        cv2.imshow('faces', result1)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            pass

        # initialize the local binary patterns descriptor along with
        # the data and label lists
        lbp = LBP_sklearn(8, 1)

        fig, axs = plt.subplots(2, 2, figsize=(80, 80), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)

        gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        gray = gray(img1)
        axs[0][0].imshow(img1, cmap='gray', vmin=0, vmax=255)
        axs[0][0].set_title('Original Image', fontdict={'fontsize': 15, 'fontweight': 'medium'})
        axs[0][0].axis('off')

        axs[0][1].imshow(gray, cmap='gray', vmin=0, vmax=255)
        axs[0][1].set_title('GrayScale Image', fontdict={'fontsize': 15, 'fontweight': 'medium'})
        axs[0][1].axis('off')
        hist = lbp.compute(gray)

        plt.show()

    def start_timer(self):      # start timer get data.
        self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.timer = QtCore.QTimer()
        if self.generate_dataset_btn.isChecked():
            self.timer.timeout.connect(self.save_dataset)
        if self.recognize_face_btn.isChecked():
            self.timer.timeout.connect(self.update_image)
        self.timer.start(5)

    def stop_timer(self):       # stop timer get data.
        self.timer.stop()
        self.ret = False
        self.capture.release()

    def update_image(self):     # update canvas image.
        if self.recognize_face_btn.isChecked():
            self.ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            faces = self.get_faces()
            self.draw_rectangle(faces)
        self.display()

    def save_dataset(self):     # Save images of new dataset generated using generate dataset button.
        location = os.path.join(self.current_path, str(self.dataset_per_subject)+".jpg")

        if self.dataset_per_subject < 1:
            QMessageBox().about(self, "Capture Berhasil", "Selanjutnya Tekan Tombol Show Image Capture \n Kemudian Recognize Expression.")
            self.generate_dataset_btn.setText("Capture")
            self.generate_dataset_btn.setChecked(False)
            self.stop_timer()
            self.dataset_per_subject = 1 # again setting max datasets

        if self.generate_dataset_btn.isChecked():
            self.ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            faces = self.get_faces()
            self.draw_rectangle(faces)
            if len(faces) is not 1:
                self.draw_text("Only One Person at a time")
            else:
                for (x, y, w, h) in faces:
                    cv2.imwrite(location, self.resize_image(self.get_gray_image()[y:y+h, x:x+w], 800, 800))
                    self.draw_text("/".join(location.split("/")[-3:]), 20, 20+ self.dataset_per_subject)
                    self.dataset_per_subject -= 1
                    self.progress_bar_generate.setValue(100 - self.dataset_per_subject*2 % 100)
        # if self.video_recording_btn.isChecked():
        #     self.recording()

        self.display()

    def display(self):      # Display in the canvas, video feed.
        pixImage = self.pix_image(self.image)
        self.video_feed.setPixmap(QtGui.QPixmap.fromImage(pixImage))
        self.video_feed.setScaledContents(True)

    def pix_image(self, image): # Converting image from OpenCv to PyQT compatible image.
        qformat = QtGui.QImage.Format_RGB888  # only RGB Image
        if len(image.shape) >= 3:
            r, c, ch = image.shape
        else:
            r, c = image.shape
            qformat = QtGui.QImage.Format_Indexed8
        pixImage = QtGui.QImage(image, c, r, image.strides[0], qformat)
        return pixImage.rgbSwapped()

    def generate(self):     # Envoke user dialog and enter name and key.
        if self.generate_dataset_btn.isChecked():
            try:
                self.current_path = os.path.join(os.getcwd(), "Capture")
                os.makedirs(self.current_path, exist_ok=True)
                self.start_timer()
                self.generate_dataset_btn.setText("Proses capture")
            except:
                msg = QMessageBox()
                msg.about(self, "User Information", '''Provide Information Please! \n name[string]\n key[integer]''')
                self.generate_dataset_btn.setChecked(False)

    def algorithm_radio_changed(self):      # When radio button change, either model is training or recognizing in respective algorithm.
        self.assign_algorithms()                                # 1. update current radio button
        self.update_recognizer()                                # 2. update face Recognizer
        self.read_model()                                       # 3. read trained data of recognizer set in step 2
        if self.train_model_btn.isChecked():
            self.train()

    def update_recognizer(self):                                # whenever algoritm radio buttons changes this function need to be invoked.
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    def assign_algorithms(self):        # Assigning anyone of algorithm to current woring algorithm.
        self.algorithm = "LBPH"

    def read_model(self):       # Reading trained model.
        if self.recognize_face_btn.isChecked():
            try:                                       # Need to to invoked when algoritm radio button change
                self.face_recognizer.read("training/"+self.algorithm.lower()+"_trained_model.yml")
            except Exception as e:
                self.print_custom_error("Unable to read Trained Model due to")
                print(e)

    def save_model(self):       # Save anyone model.
        try:
            self.face_recognizer.save("training/"+self.algorithm.lower()+"_trained_model.yml")
            msg = self.algorithm+" model trained, stop training or train another model"
            self.trained_model += 1
            self.progress_bar_train.setValue(self.trained_model)
            QMessageBox().about(self, "Training Completed", msg)
        except Exception as e:
            self.print_custom_error("Unable to save Trained Model due to")
            print(e)

    def train(self):        # When train button is clicked.
        if self.train_model_btn.isChecked():
            button = self.algo_radio_group.checkedButton()
            button.setEnabled(False)
            self.train_model_btn.setText("Stop Training")
            os.makedirs("training", exist_ok=True)
            labels, faces = self.get_labels_and_faces()
            try:
                msg = self.algorithm+" model training started"
                QMessageBox().about(self, "Training Started", msg)

                self.face_recognizer.train(faces, np.array(labels))
                self.save_model()
            except Exception as e:
                self.print_custom_error("Unable To Train the Model Due to: ")
                print(e)
        else:
            # self.eigen_algo_radio.setEnabled(True)
            # self.fisher_algo_radio.setEnabled(True)
            self.lbph_algo_radio.setEnabled(True)
            self.train_model_btn.setChecked(False)
            self.train_model_btn.setText("Train Model")

    def recognize(self):        # When recognized button is called.
        if self.recognize_face_btn.isChecked():
            self.start_timer()
            self.recognize_face_btn.setText("Stop Recognition")
            self.read_model()
        else:
            self.recognize_face_btn.setText("Recognize Expression")
            self.stop_timer()

    def get_all_key_name_pairs(self):       # Get all (key, name) pair of datasets present in datasets.
        return dict([subfolder.split('-') for _, folders, _ in os.walk(os.path.join(os.getcwd(), "datasets")) for subfolder in folders],)

    def absolute_path_generator(self):      # Generate all path in dataset folder.
        separator = "-"
        for folder, folders, _ in os.walk(os.path.join(os.getcwd(),"datasets")):
            for subfolder in folders:
                subject_path = os.path.join(folder,subfolder)
                key, _ = subfolder.split(separator)
                for image in os.listdir(subject_path):
                    absolute_path = os.path.join(subject_path, image)
                    yield absolute_path,key

    def get_labels_and_faces(self):     # Get label and faces.
        labels, faces = [],[]
        for path,key in self.absolute_path_generator():
            faces.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
            labels.append(int(key))
        return labels,faces

    def get_gray_image(self):       # Convert BGR image to GRAY image.
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def get_faces(self):        # Get all faces in a image.
        # variables
        scale_factor = 1.1
        min_neighbors = 8
        min_size = (100, 100)

        faces = self.face_classifier.detectMultiScale(
                            self.get_gray_image(),
                            scaleFactor = scale_factor,
                            minNeighbors = min_neighbors,
                            minSize = min_size)

        return faces

    def get_smiles(self, roi_gray):     # Get all smiles in a image.
        scale_factor = 1.7
        min_neighbors = 22
        min_size = (25, 25)
        #window_size = (200, 200)

        smiles = self.smile_classifier.detectMultiScale(
                            roi_gray,
                            scaleFactor = scale_factor,
                            minNeighbors = min_neighbors,
                            minSize = min_size
                            )

        return smiles

    def get_eyes(self, roi_gray):       # Get all eyes in a image.
        scale_factor = 1.1
        min_neighbors = 6
        min_size = (30, 30)

        eyes = self.eye_classifier.detectMultiScale(
                            roi_gray,
                            scaleFactor = scale_factor,
                            minNeighbors = min_neighbors,
                            #minSize = min_size
                            )

        return eyes

    def draw_rectangle(self, faces):        # Draw rectangle either in face, eyes or smile.
        for (x, y, w, h) in faces:
            roi_gray_original = self.get_gray_image()[y:y + h, x:x + w]
            roi_gray = self.resize_image(roi_gray_original, 92, 112)
            roi_color = self.image[y:y+h, x:x+w]
            if self.recognize_face_btn.isChecked():
                try:
                    predicted, confidence = self.face_recognizer.predict(roi_gray)
                    name = self.get_all_key_name_pairs().get(str(predicted))
                    self.draw_text("Pengenalan Menggunakan: "+self.algorithm, 70,50)
                    if self.lbph_algo_radio.isChecked():
                        if confidence > 105:
                            msg = "Mendekati [" + name + "]"
                        else:
                            confidence = "{:.2f}".format(100 - confidence)
                            msg = name
                        self.progress_bar_recognize.setValue(float(confidence))
                    else:
                        msg = name
                        self.progress_bar_recognize.setValue(int(confidence%100))
                        confidence = "{:.2f}".format(confidence)

                    self.draw_text(msg, x-5,y-5)
                except Exception as e:
                    self.print_custom_error("Tidak bisa mengenali karena")
                    print(e)

            if self.eye_rect_radio.isChecked():     # If eye radio button is checked.
                eyes = self.get_eyes(roi_gray_original)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            elif self.smile_rect_radio.isChecked():     # If smile radio button is checked.
                smiles = self.get_smiles(roi_gray_original)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            else:       # If face radio button is checked.
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def time(self):     # Get current time.
        return datetime.now().strftime("%d-%b-%Y:%I-%M-%S")

    def draw_text(self, text, x=20, y=20, font_size=2, color = (0, 255, 0)): # Draw text in current image in particular color.
        cv2.putText(self.image, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.6, color, font_size)

    def resize_image(self, image, width=280, height=280): # Resize image before storing.
        return cv2.resize(image, (width,height), interpolation = cv2.INTER_CUBIC)

    def print_custom_error(self, msg):      # Print custom error message/
        print("="*100)
        print(msg)
        print("="*100)

    def recording(self):        # Record Video when either recognizing or generating.
        if self.ret:
            self.video_output.write(self.image)

    def about_info(self):       # Menu Information of info button of application.
        msg_box = QMessageBox()
        msg_box.setText('''
            FacialExpression (authenticate using face recognition) is an Python/OpenCv based
            face recognition application. It uses Machine Learning to train the
            model generated using haar classifier.
            LBPH algorithms are implemented.
        ''')
        msg_box.setInformativeText('''
           INSTITUT TEKNOLOGI NASIONAL BANDUNG
            ''')
        msg_box.setWindowTitle("About FacialExpression")
        msg_box.exec_()

    def help_info(self):       # Menu Information of help button of application.
        msg_box = QMessageBox()
        msg_box.setText('''
            This application is capable of creating datasets, generating models,
            recording videos and clicking images.
            Detection of face, eyes, smile are also implemented.
            Recognition of person is primary job of this application.
        ''')
        msg_box.setInformativeText('''
            Follow these steps to use this application
            1. Generate atlest two datasets.
            2. Train all algoritmic model using given radio buttons.
            3. Recognize person.

            ''')
        msg_box.setWindowTitle("Help")
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = FacialExpression()         # Running application loop.
    ui.show()
    sys.exit(app.exec_())       #  Exit application.
