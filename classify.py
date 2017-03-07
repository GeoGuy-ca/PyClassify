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
import os

class Sensor:
    LANDSAT_8 = 1
    #TODO Add more platforms

Ui_MainWindow, QtBaseClass = uic.loadUiType("classifyGUI.ui")
filePath = None
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

    def openfilestart(self):
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
        filePath = '/home/mmclean/LC80490222016230LGN00/LC80490222016230LGN00_B1.TIF'
        if filePath[-28:-25] == 'LC8':
            platform = Sensor.LANDSAT_8
            data_set = gdal.Open(str(filePath[:-6] + 'B1.TIF'), GA_ReadOnly)
            band1 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B2.TIF'), GA_ReadOnly)
            band2 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B3.TIF'), GA_ReadOnly)
            band3 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B4.TIF'), GA_ReadOnly)
            band4 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B5.TIF'), GA_ReadOnly)
            band5 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B6.TIF'), GA_ReadOnly)
            band6 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B7.TIF'), GA_ReadOnly)
            band7 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B9.TIF'), GA_ReadOnly)
            band9 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B10.TIF'), GA_ReadOnly)
            band10 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B11.TIF'), GA_ReadOnly)
            band11 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            print band1[2000, 2500]
        else:
            print 'only LC8 currently supported!'
        self.file_path.setText(filePath)



        y_size = int(data_set.RasterYSize)
        x_size = int(data_set.RasterXSize)

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
        filePath = QFileDialog.getOpenFileName()
        if  filePath[-28:-25] == 'LC8':
            platform = Sensor.LANDSAT_8
            data_set = gdal.Open(str(filePath[:-6] + 'B1.TIF'), GA_ReadOnly)
            band1 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B2.TIF'), GA_ReadOnly)
            band2 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B3.TIF'), GA_ReadOnly)
            band3 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B4.TIF'), GA_ReadOnly)
            band4 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B5.TIF'), GA_ReadOnly)
            band5 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B6.TIF'), GA_ReadOnly)
            band6 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B7.TIF'), GA_ReadOnly)
            band7 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B9.TIF'), GA_ReadOnly)
            band9 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B10.TIF'), GA_ReadOnly)
            band10 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            data_set = gdal.Open(str(filePath[:-6] + 'B11.TIF'), GA_ReadOnly)
            band11 = np.array(data_set.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            print band1[2000, 2500]
        else:
            print 'only LC8 currently supported!'
        self.file_path.setText(filePath)



        y_size = int(data_set.RasterYSize)
        x_size = int(data_set.RasterXSize)

    def k_means(self):
        global dev_data
        global result
        global stream
        global preGPU
        global x_size
        global y_size
        num_clust = 5
        max_iteration = 20
        print x_size
        print y_size
        clusters = np.sort(np.ndarray([num_clust+1, preGPU.shape[2]], dtype=np.uint16)) #Nuber of clusters by number of bands
        for i in range(1, num_clust + 1):
            x = random.randint(0, x_size - 1)
            y = random.randint(0, y_size - 1)
            clusters[i, :] = preGPU[x, y, :]
            while clusters[i, 0] == 0:
                x = random.randint(0, x_size-1)
                y = random.randint(0, y_size-1)
                clusters[i, :] = preGPU[x, y, :]

        dev_clusters = cuda.to_device(clusters, stream=stream)
        result = np.ndarray([x_size, y_size], dtype=np.uint8)
        dev_result = cuda.device_array([x_size, y_size], dtype=np.uint8, stream=stream)

        dev_data = cuda.to_device(preGPU, stream=stream)
        stream.synchronize()
        start = time.time()
        k_means.k_means_classify[(x_size*y_size)/1024+1, 1024, stream](dev_data, dev_result, y_size, dev_clusters, num_clust, preGPU.shape[2], max_iteration)
        stream.synchronize()
        print "K-Means: " + str(time.time() - start) + " Seconds"
        start = time.time()
        dev_result.copy_to_host(result, stream=stream)
        stream.synchronize()
        print "Transfer result to host in " + str(time.time() - start) + " Seconds"
        # for i in range(3000, 3010):
        #     for j in range(3000, 3010):
        #         print result[i, j]
        test = np.empty([3, 3])
        try:
            pixmap.fill(3)
        except:
            print "Pixmap not Def"
        pixmap = QImage(result, x_size, y_size, y_size, QImage.Format_Indexed8)
        pixmap.setColor(0, qRgb(0, 0, 0))
        pixmap.setColor(1, qRgb(57, 255, 139))
        pixmap.setColor(2, qRgb(57, 16, 13))
        pixmap.setColor(3, qRgb(252, 216, 213))
        pixmap.setColor(4, qRgb(152, 116, 113))
        self.preview.setPixmap(QPixmap.fromImage(pixmap))


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
        if self.select_B1.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band1[:, :, np.newaxis], dtype=np.uint16)
            else:
                preGPU = np.dstack((preGPU, band1))
        if self.select_B2.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band2[:, :, np.newaxis], dtype=int)
            else:
                preGPU = np.dstack((preGPU, band2))
        if self.select_B3.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band3[:, :, np.newaxis], dtype=int)
            else:
                preGPU = np.dstack((preGPU, band3))
        if self.select_B4.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band4[:, :, np.newaxis], dtype=int)
            else:
                preGPU = np.dstack((preGPU, band4))
        if self.select_B5.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band5[:, :, np.newaxis], dtype=int)
            else:
                preGPU = np.dstack((preGPU, band5))
        if self.select_B6.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band6[:, :, np.newaxis], dtype=int)
            else:
                preGPU = np.dstack((preGPU, band6))
        if self.select_B7.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band7[:, :, np.newaxis], dtype=int)
            else:
                preGPU = np.dstack((preGPU, band7))
        if self.select_B8.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band8[:, :, np.newaxis], dtype=int)
            else:
                preGPU = np.dstack((preGPU, band8))
        if self.select_B9.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band9[:, :, np.newaxis], dtype=int)
            else:
                preGPU = np.dstack((preGPU, band9))
        if self.select_B10.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band10[:, :, np.newaxis], dtype=int)
            else:
                preGPU = np.dstack((preGPU, band10))
        if self.select_B11.isChecked():
            if preGPU is None:
                preGPU = np.ascontiguousarray(band11[:, :, np.newaxis], dtype=int)
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
    window.openfilestart()
    window.loadGPU()
    window.k_means()
    print "Program Initialized"
    sys.exit(app.exec_())




