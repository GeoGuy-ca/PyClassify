import sys
from numbapro import cuda

import gdal
from gdalconst import *
import PyQt4.QtGui as QtGui
import PyQt4.uic as uic
import numpy as np
import k_means
import time


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


class MyApp(QtGui.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.open_image_button.clicked.connect(self.openfile)
        self.load_gpu_button.clicked.connect(self.loadGPU)
        self.k_means_button.clicked.connect(self.k_means)

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
        filePath = QtGui.QFileDialog.getOpenFileName()
        if  filePath[-28:-25] == 'LC8':
            platform = Sensor.LANDSAT_8
            data_set = gdal.Open(str(filePath[:-6] + 'B1.TIF'), GA_ReadOnly)
            band1 = np.array(data_set.GetRasterBand(1).ReadAsArray())
            data_set = gdal.Open(str(filePath[:-6] + 'B2.TIF'), GA_ReadOnly)
            band2 = np.array(data_set.GetRasterBand(1).ReadAsArray())
            data_set = gdal.Open(str(filePath[:-6] + 'B3.TIF'), GA_ReadOnly)
            band3 = np.array(data_set.GetRasterBand(1).ReadAsArray())
            data_set = gdal.Open(str(filePath[:-6] + 'B4.TIF'), GA_ReadOnly)
            band4 = np.array(data_set.GetRasterBand(1).ReadAsArray())
            data_set = gdal.Open(str(filePath[:-6] + 'B5.TIF'), GA_ReadOnly)
            band5 = np.array(data_set.GetRasterBand(1).ReadAsArray())
            data_set = gdal.Open(str(filePath[:-6] + 'B6.TIF'), GA_ReadOnly)
            band6 = np.array(data_set.GetRasterBand(1).ReadAsArray())
            data_set = gdal.Open(str(filePath[:-6] + 'B7.TIF'), GA_ReadOnly)
            band7 = np.array(data_set.GetRasterBand(1).ReadAsArray())
            data_set = gdal.Open(str(filePath[:-6] + 'B9.TIF'), GA_ReadOnly)
            band9 = np.array(data_set.GetRasterBand(1).ReadAsArray())
            data_set = gdal.Open(str(filePath[:-6] + 'B10.TIF'), GA_ReadOnly)
            band10 = np.array(data_set.GetRasterBand(1).ReadAsArray())
            data_set = gdal.Open(str(filePath[:-6] + 'B11.TIF'), GA_ReadOnly)
            band11 = np.array(data_set.GetRasterBand(1).ReadAsArray())
        else:
            print 'only LC8 currently supported!'
        self.file_path.setText(filePath)



        x_size = int(data_set.RasterXSize)
        y_size = int(data_set.RasterYSize)

    def k_means(self):
        global dev_data
        global result
        global stream
        global preGPU
        if len(preGPU.shape) == 3:
            dev_result = cuda.device_array_like(preGPU[:, :, 0], stream=stream)
        else:
            dev_result = cuda.device_array_like(preGPU[:, :], stream=stream)
        k_means.k_means_classify(dev_data, dev_result, 5, 20)
        start = time.time()
        dev_result.copy_to_host(result, stream=stream)
        print "Transfer result to host in " + str(time.time() - start) + " Seconds"
        print result[50, 50]

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
                preGPU = np.ascontiguousarray(band1[:, :, np.newaxis], dtype=int)
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
        else:
            print "Failed to load GPU"


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    print "Program Initialized"
    sys.exit(app.exec_())




