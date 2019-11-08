from PyQt5.QtCore import Qt
import random
import sys

import matplotlib
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import arange, pi, sin
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QListView, QAction, QApplication, QMainWindow, QMenu,
                             QMessageBox, QFileDialog, QSizePolicy, QHBoxLayout, QWidget)
from PyQt5.QtCore import QStringListModel, QAbstractListModel, QModelIndex, QSize
import h5py
import numpy as np

matplotlib.use("Qt5Agg")


class MyMplCanvas(FigureCanvas):
    """这是一个窗口部件，即QWidget（当然也是FigureCanvasAgg）"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """静态画布：一条正弦线"""

    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)


class MyDynamicMplCanvas(MyMplCanvas):
    """动态画布：每秒自动更新，更换一条折线。"""

    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # 构建4个随机整数，位于闭区间[0, 10]
        l = [random.randint(0, 10) for i in range(4)]

        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()


class ApplicationWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("skeleton display")

        # top-level menu
        self.file_menu = QMenu('&File', self)
        self.help_menu = QMenu('&Help', self)

        # list view
        listModel = QStringListModel()
        # listModel.setStringList(['items'])
        listView = QListView()
        listView.setModel(listModel)

        # menu action
        self.file_menu.addAction(
            '&Import Features', lambda: self.importFeatures(listModel), QtCore.Qt.CTRL + QtCore.Qt.Key_F)
        self.file_menu.addAction(
            '&Import Labels', lambda: self.importLabels(listModel), QtCore.Qt.CTRL + QtCore.Qt.Key_L)
        self.file_menu.addAction(
            '&Import Models', lambda: self.importModels(listModel), QtCore.Qt.CTRL + QtCore.Qt.Key_M)
        self.file_menu.addAction('&Quit', self.quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.help_menu.addAction('&About', self.about)

        # add menu
        self.menuBar().addMenu(self.file_menu)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        # window
        self.main_widget = QWidget(self)

        # layout = QVBoxLayout(self.main_widget)
        layout = QHBoxLayout(self.main_widget)
        layout.setDirection(QHBoxLayout.LeftToRight)
        layout.addWidget(listView, 0, Qt.AlignLeft)
        layout.addWidget(MyDynamicMplCanvas(
            self.main_widget, width=5, height=4, dpi=100), 0, Qt.AlignRight)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        # 状态条显示2秒
        self.statusBar().showMessage("skeleton display program", 2000)

    def importFeatures(self, listModel):
        fname = QFileDialog.getOpenFileName(self, 'open feature file', '/')
        if fname[0] is not None:
            f = h5py.File(fname[0], 'r')
            features = np.array([f[element]
                                 for element in np.squeeze(f['features'][:])])
            n_features = features.shape[0]
            idx = np.arange(n_features).astype(str)
            listModel.setStringList(idx)
            
    def importLabels(self, listModel):
        fname = QFileDialog.getOpenFileName(self, 'open label file', '/')
        if fname[0] is not None:
            f = h5py.File(fname[0], 'r')
            features = np.array([f[element]
                                 for element in np.squeeze(f['features'][:])])
            n_features = features.shape[0]
            idx = np.arange(n_features).astype(str)
            listModel.setStringList(idx)

    def importModels(self, listModel):
        fname = QFileDialog.getOpenFileName(self, 'open model file', '/')
        if fname[0] is not None:
            f = h5py.File(fname[0], 'r')
            features = np.array([f[element]
                                 for element in np.squeeze(f['features'][:])])
            n_features = features.shape[0]
            idx = np.arange(n_features).astype(str)
            listModel.setStringList(idx)

    def quit(self):
        self.close()

    def closeEvent(self, ce):
        self.quit()

    def about(self):
        QMessageBox.about(self, "About",
                          """embedding_in_qt5.py example
        Copyright 2015 BoxControL

        This program is a simple example of a Qt5 application embedding matplotlib
        canvases. It is base on example from matplolib documentation, and initially was
        developed from Florent Rougon and Darren Dale.

        http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html

        It may be used and modified with no restriction; raw copies as well as
        modified versions may be distributed without limitation.
        """
                          )


if __name__ == '__main__':
    app = QApplication(sys.argv)

    aw = ApplicationWindow()
    aw.show()
    # sys.exit(qApp.exec_())
    app.exec_()
