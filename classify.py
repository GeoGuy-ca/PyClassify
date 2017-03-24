import sys
from numbapro import cuda

import gdal
from gdalconst import *
from PyQt4.QtGui import *
import PyQt4.uic as uic
import numpy as np
import k_means
import time
import random


class Sensor:
    LANDSAT_8 = 1
    #TODO Add more platforms

Ui_MainWindow, QtBaseClass = uic.loadUiType("classifyGUI.ui")
filePath = None
num_clust = None
band1 = None
band2 = None
band3 = None
band4 = None
band5 = None
band6 = None
band7 = None
band8 = None
band9 = None
band10 = None
band11 = None
preGPU = None
dev_data = None
data_set = None
result = None
stream = cuda.stream()
x_size = 0
y_size = 0
original_image = None
platform = None


class MyApp(QMainWindow, Ui_MainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.open_image_button.clicked.connect(self.openfile)
        self.load_gpu_button.clicked.connect(self.loadGPU)
        self.k_means_button.clicked.connect(self.k_means)
        self.k_means_no_sqrt.clicked.connect(self.k_means_no_root)
        self.k_means_button_cpu.clicked.connect(self.k_means_seq)
        self.k_means_button_cpu_no_sqrt.clicked.connect(self.k_means_seq_no_root)
        self.fuzzy_k_means_button.clicked.connect(self.k_means_fuzzy)
        self.fuzz_k_means_button_cpu.clicked.connect(self.k_means_fuzzy_seq)
        self.save_button.clicked.connect(self.save)


    def save(self):
        global result
        global filePath
        global x_size
        global y_size
        global data_set
        outpath = QFileDialog.getSaveFileName(self, "saveFlle", filePath[:-4] + "_class.tiff", filter="tiff (*.tiff)")
        print outpath
        output = gdal.GetDriverByName('GTiff').Create(str(outpath), x_size, y_size, 1, gdal.GDT_Byte)
        output.SetGeoTransform(data_set.GetGeoTransform())
        output.SetProjection(data_set.GetProjection())
        output.GetRasterBand(1).WriteArray(result)
        output.FlushCache()

    def openfile(self):
        global band1
        global band2
        global band3
        global band4
        global band5
        global band6
        global band7
        global band8
        global band9
        global band10
        global band11
        global platform
        global x_size
        global y_size
        global filePath
        global data_set
        filePath = QFileDialog.getOpenFileName()
        if  filePath[-28:-25] == 'LC8' or filePath[-10:-7] =='LC8':
            platform = Sensor.LANDSAT_8
            data_set = gdal.Open(str(filePath[:-6] + 'B1.TIF'), GA_ReadOnly)
            band1 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B1.setEnabled(True)
            data_set = gdal.Open(str(filePath[:-6] + 'B2.TIF'), GA_ReadOnly)
            band2 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B2.setEnabled(True)
            data_set = gdal.Open(str(filePath[:-6] + 'B3.TIF'), GA_ReadOnly)
            band3 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B3.setEnabled(True)
            data_set = gdal.Open(str(filePath[:-6] + 'B4.TIF'), GA_ReadOnly)
            band4 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B4.setEnabled(True)
            data_set = gdal.Open(str(filePath[:-6] + 'B5.TIF'), GA_ReadOnly)
            band5 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B5.setEnabled(True)
            data_set = gdal.Open(str(filePath[:-6] + 'B6.TIF'), GA_ReadOnly)
            band6 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B6.setEnabled(True)
            data_set = gdal.Open(str(filePath[:-6] + 'B7.TIF'), GA_ReadOnly)
            band7 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B7.setEnabled(True)
            self.select_B8.setEnabled(False)
            data_set = gdal.Open(str(filePath[:-6] + 'B9.TIF'), GA_ReadOnly)
            band9 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B9.setEnabled(True)
            data_set = gdal.Open(str(filePath[:-6] + 'B10.TIF'), GA_ReadOnly)
            band10 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B10.setEnabled(True)
            data_set = gdal.Open(str(filePath[:-6] + 'B11.TIF'), GA_ReadOnly)
            band11 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B11.setEnabled(True)
        else:
            data_set = gdal.Open(str(filePath), GA_ReadOnly)
            band1 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            self.select_B1.setEnabled(True)
            band2 = np.array(data_set.GetRasterBand(2).ReadAsArray(), dtype=np.uint16)
            self.select_B2.setEnabled(True)
            band3 = np.array(data_set.GetRasterBand(3).ReadAsArray(), dtype=np.uint16)
            self.select_B3.setEnabled(True)
            self.select_B4.setEnabled(False)
            self.select_B5.setEnabled(False)
            self.select_B6.setEnabled(False)
            self.select_B7.setEnabled(False)
            self.select_B8.setEnabled(False)
            self.select_B9.setEnabled(False)
            self.select_B10.setEnabled(False)
            self.select_B11.setEnabled(False)
        self.file_path.setText(filePath)



        y_size = int(data_set.RasterYSize)
        x_size = int(data_set.RasterXSize)


    def k_means_seq(self):
        global result
        global stream
        global preGPU
        global x_size
        global y_size
        global num_clust
        num_clust = int(self.classes.text())
        max_iteration = int(self.itterations.text())
        clusters = np.ndarray([num_clust + 1, preGPU.shape[2]], dtype=np.uint16)  # Nuber of clusters by number of bands
        for i in range(1, num_clust + 1):
            x = random.randint(0, preGPU.shape[1] - 1)
            y = random.randint(0, preGPU.shape[0] - 1)
            clusters[i, :] = preGPU[y, x, :]
            while clusters[i, 0] == 0:
                x = random.randint(0, preGPU.shape[1] - 1)
                y = random.randint(0, preGPU.shape[0] - 1)
                clusters[i, :] = preGPU[y, x, :]
        result = np.ndarray([y_size, x_size], dtype=np.uint8)
        start = time.time()
        k_means.k_means_classify_seq(preGPU, result, x_size, y_size, clusters, num_clust, preGPU.shape[2], max_iteration, self.debug.isChecked())
        print "K-Means CPU: " + str(time.time() - start) + " Seconds"
        self.show_result()

    def k_means_seq_no_root(self):
        global result
        global stream
        global preGPU
        global x_size
        global y_size
        global num_clust
        num_clust = int(self.classes.text())
        max_iteration = int(self.itterations.text())
        clusters = np.ndarray([num_clust + 1, preGPU.shape[2]], dtype=np.uint16)  # Nuber of clusters by number of bands
        for i in range(1, num_clust + 1):
            x = random.randint(0, preGPU.shape[1] - 1)
            y = random.randint(0, preGPU.shape[0] - 1)
            clusters[i, :] = preGPU[y, x, :]
            while clusters[i, 0] == 0:
                x = random.randint(0, preGPU.shape[1] - 1)
                y = random.randint(0, preGPU.shape[0] - 1)
                clusters[i, :] = preGPU[y, x, :]
        result = np.ndarray([y_size, x_size], dtype=np.uint8)
        start = time.time()
        k_means.k_means_classify_seq_no_root(preGPU, result, x_size, y_size, clusters, num_clust, preGPU.shape[2], max_iteration, self.debug.isChecked())
        print "K-Means CPU (NO SQRT): " + str(time.time() - start) + " Seconds"
        self.show_result()

    def k_means_fuzzy_seq(self):
        global result
        global stream
        global preGPU
        global x_size
        global y_size
        global num_clust
        num_clust = int(self.classes.text())
        max_iteration = int(self.itterations.text())
        clusters = np.ndarray([num_clust + 1, preGPU.shape[2]],
                              dtype=np.uint16)  # Nuber of clusters by number of bands
        for i in range(1, num_clust + 1):
            x = random.randint(0, preGPU.shape[1] - 1)
            y = random.randint(0, preGPU.shape[0] - 1)
            clusters[i, :] = preGPU[y, x, :]
            while clusters[i, 0] == 0:
                x = random.randint(0, preGPU.shape[1] - 1)
                y = random.randint(0, preGPU.shape[0] - 1)
                clusters[i, :] = preGPU[y, x, :]
        result = np.ndarray([y_size, x_size], dtype=np.uint8)
        start = time.time()
        k_means.k_means_classify_fuzzy_seq(preGPU, result, x_size, y_size, clusters, num_clust, preGPU.shape[2], max_iteration, float(self.movement.text()), self.debug.isChecked())
        print "Fuzzy K-Means CPU: " + str(time.time() - start) + " Seconds"
        self.show_result()

    def k_means(self):
        global dev_data
        global result
        global stream
        global preGPU
        global x_size
        global y_size
        global num_clust
        num_clust = int(self.classes.text())
        max_iteration = int(self.itterations.text())
        clusters = np.ndarray([num_clust+1, preGPU.shape[2]], dtype=np.uint16) #Nuber of clusters by number of bands
        for i in range(1, num_clust + 1):
            x = random.randint(0, preGPU.shape[1] - 1)
            y = random.randint(0, preGPU.shape[0] - 1)
            clusters[i, :] = preGPU[y, x, :]
            while clusters[i, 0] == 0:
                x = random.randint(0, preGPU.shape[1] - 1)
                y = random.randint(0, preGPU.shape[0] - 1)
                clusters[i, :] = preGPU[y, x, :]

        dev_clusters = cuda.to_device(clusters, stream=stream)
        result = np.ndarray([y_size, x_size], dtype=np.uint8)
        dev_result = cuda.device_array([y_size, x_size], dtype=np.uint8, stream=stream)

        dev_data = cuda.to_device(preGPU, stream=stream)
        stream.synchronize()
        #print clusters
        start = time.time()
        k_means.k_means_classify(dev_data, dev_result, x_size, y_size, dev_clusters, num_clust, preGPU.shape[2], max_iteration, self.debug.isChecked(), stream)

        stream.synchronize()
        print "K-Means GPU: " + str(time.time() - start) + " Seconds"
        start = time.time()
        dev_clusters.copy_to_host(clusters, stream=stream)
        #print clusters
        dev_result.copy_to_host(result, stream=stream)
        stream.synchronize()
        print "Transfer result to host in " + str(time.time() - start) + " Seconds"
        self.show_result()


    def k_means_no_root(self):
        global dev_data
        global result
        global stream
        global preGPU
        global x_size
        global y_size
        global num_clust
        num_clust = int(self.classes.text())
        max_iteration = int(self.itterations.text())
        clusters = np.ndarray([num_clust + 1, preGPU.shape[2]],
                              dtype=np.uint16)  # Nuber of clusters by number of bands
        for i in range(1, num_clust + 1):
            x = random.randint(0, preGPU.shape[1] - 1)
            y = random.randint(0, preGPU.shape[0] - 1)
            clusters[i, :] = preGPU[y, x, :]
            while clusters[i, 0] == 0:
                x = random.randint(0, preGPU.shape[1] - 1)
                y = random.randint(0, preGPU.shape[0] - 1)
                clusters[i, :] = preGPU[y, x, :]

        dev_clusters = cuda.to_device(clusters, stream=stream)
        result = np.ndarray([y_size, x_size], dtype=np.uint8)
        dev_result = cuda.device_array([y_size, x_size], dtype=np.uint8, stream=stream)

        dev_data = cuda.to_device(preGPU, stream=stream)
        stream.synchronize()
        # print clusters
        start = time.time()
        k_means.k_means_classify_no_root(dev_data, dev_result, x_size, y_size, dev_clusters, num_clust, preGPU.shape[2],
                                 max_iteration, self.debug.isChecked(), stream)

        stream.synchronize()
        print "K-Means GPU (NO SQRT): " + str(time.time() - start) + " Seconds"
        start = time.time()
        dev_clusters.copy_to_host(clusters, stream=stream)
        # print clusters
        dev_result.copy_to_host(result, stream=stream)
        stream.synchronize()
        print "Transfer result to host in " + str(time.time() - start) + " Seconds"
        self.show_result()

    def k_means_fuzzy(self):
        global dev_data
        global result
        global stream
        global preGPU
        global x_size
        global y_size
        global num_clust
        num_clust = int(self.classes.text())
        max_iteration = int(self.itterations.text())
        clusters = np.ndarray([num_clust+1, preGPU.shape[2]], dtype=np.uint16) #Nuber of clusters by number of bands
        for i in range(1, num_clust + 1):
            x = random.randint(0, preGPU.shape[1] - 1)
            y = random.randint(0, preGPU.shape[0] - 1)
            clusters[i, :] = preGPU[y, x, :]
            while clusters[i, 0] == 0:
                x = random.randint(0, preGPU.shape[1] - 1)
                y = random.randint(0, preGPU.shape[0] - 1)
                clusters[i, :] = preGPU[y, x, :]

        dev_clusters = cuda.to_device(clusters, stream=stream)
        result = np.ndarray([y_size, x_size], dtype=np.uint8)
        dev_result = cuda.device_array([y_size, x_size], dtype=np.uint8, stream=stream)

        dev_data = cuda.to_device(preGPU, stream=stream)
        stream.synchronize()
        #print clusters
        start = time.time()
        k_means.k_means_classify_fuzzy(dev_data, dev_result, x_size, y_size, dev_clusters, num_clust, preGPU.shape[2], max_iteration, self.debug.isChecked(), float(self.movement.text()), stream)

        stream.synchronize()
        print "Fuzzy K-Means GPU: " + str(time.time() - start) + " Seconds"
        start = time.time()
        dev_clusters.copy_to_host(clusters, stream=stream)
        #print clusters
        dev_result.copy_to_host(result, stream=stream)
        stream.synchronize()
        print "Transfer result to host in " + str(time.time() - start) + " Seconds"
        self.show_result()


    def show_result(self):
        global result
        global x_size
        global y_size
        global num_clust
        pixmap = QImage(result, x_size, min(8000, y_size), x_size, QImage.Format_Indexed8)
        colormap = []

        for i in range(0, num_clust):
            colormap.append(qRgb((200*i) % 255, (120*i) % 255, (30*i) % 255))
        pixmap.setColorTable(colormap)
        self.preview.resize(x_size, min(8000, y_size))
        self.image_area.resize(x_size, min(8000, y_size))
        self.preview.setPixmap(QPixmap.fromImage(pixmap))

    #This method checks which band boxes are checked and stacks selected bands into a contiguous array and passes it to the GPU
    #Note that the preGUP array is initalized ascontiguous, and np.uint16, DO NOT USE non numpy datatypes!!!!
    def loadGPU(self):
        global band1
        global band2
        global band3
        global band4
        global band5
        global band6
        global band7
        global band8
        global band9
        global band10
        global band11
        global dev_data
        global preGPU
        global stream
        preGPU = None
        if self.select_B1.isChecked() and self.select_B1.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band1[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band1))
        if self.select_B2.isChecked() and self.select_B2.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band2[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band2))
        if self.select_B3.isChecked() and self.select_B3.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band3[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band3))
        if self.select_B4.isChecked() and self.select_B4.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band4[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band4))
        if self.select_B5.isChecked()  and self.select_B5.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band5[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band5))
        if self.select_B6.isChecked() and self.select_B6.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band6[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band6))
        if self.select_B7.isChecked() and self.select_B7.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band7[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band7))
        if self.select_B8.isChecked() and self.select_B8.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band8[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band8))
        if self.select_B9.isChecked() and self.select_B9.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band9[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band9))
        if self.select_B10.isChecked() and self.select_B10.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band10[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band10))
        if self.select_B11.isChecked() and self.select_B11.isEnabled():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band11[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band11))
        if preGPU is not None:
            print preGPU.shape
            start = time.time()
            dev_data = cuda.to_device(preGPU, stream=stream)
            print "Transfer to GPU in " + str(time.time() - start) + " Seconds"
            # pixmap = QImage(preGPU, y_size, x_size, QImage.Format_RGB32)
            # self.preview.setPixmap(QPixmap.fromImage(pixmap))
        else:
            print "Failed to load GPU"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    print "Program Initialized"
    sys.exit(app.exec_())




