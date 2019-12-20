import json
import os
import random
import sys
import numba
import h5py
import matplotlib
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy import arange, pi, sin
from PyQt5 import QtCore
from PyQt5.QtCore import (QAbstractListModel, QModelIndex, QSize,
                          QStringListModel, Qt)
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QGridLayout,
                             QHBoxLayout, QLabel, QListView, QMainWindow,
                             QMenu, QMessageBox, QSizePolicy, QWidget)

from rgb_mapping import classification_pipline

matplotlib.use("Qt5Agg")


class StringLabeler(object):
    def __init__(self, ldict):
        assert type(ldict) is dict

        values = list(ldict.values())
        keys = list(ldict.keys())
        self.content = np.char.asarray(values[0])
        self.content = keys[0]+': ' + \
            np.char.asarray(values[0]) + \
            '\t'

        for i in range(1, len(keys)):
            label = keys[i]
            self.content = self.content + label
            self.content = self.content + ': ' + \
                np.char.asarray(ldict[label].astype(str)) + \
                '\t'

    def addItems(self, ldict):
        for k in ldict:
            self.content = self.content + k
            self.content = self.content + ': ' + \
                np.char.asarray(ldict[k].astype(str)) + \
                '\t'

    def addSingleItem(self, ldict, idx):
        for k in ldict:
            self.content[idx] = self.content[idx]+k
            self.content[idx] = self.content[idx] + ': ' + \
                np.char.asarray(ldict[k].astype(str)) + \
                '\t'

    def clear(self):
        self.content = np.char.chararray(self.content.size)

    def getContent(self):
        return self.content


class Canvas(FigureCanvas):
    """这是一个窗口部件，即QWidget（当然也是FigureCanvasAgg）"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = p3.Axes3D(fig, azim=-90, elev=10)
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


class DynamicCanvas(Canvas):
    """动态画布：每秒自动更新"""

    def __init__(self, *args, **kwargs):
        Canvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(100)

    def compute_initial_figure(self):
        self.points = self.axes.scatter([], [], [])

        self.fcounter = 0
        self.axes.set_xlim3d([1, 3])
        self.axes.set_xlabel('X')

        self.axes.set_ylim3d([1, 3])
        self.axes.set_ylabel('Z')

        self.axes.set_zlim3d([1, 3])
        self.axes.set_zlabel('Y')
        self.lc = Line3DCollection(segments=[])
        self.axes.add_collection3d(self.lc)

        f = json.load(open('data/MSRAction3D/body_model.json', 'r'))
        self.bones = np.array(f['bones']) - 1

        self.video = None

    def update_figure(self):
        if self.video is None:
            return
        if self.fcounter >= self.video.shape[0]:
            self.fcounter = self.video.shape[0]-1

        frame = self.video[self.fcounter]+2
        self.points._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])

        lines = [np.array([frame[self.bones[i, 0]], frame[self.bones[i, 1]]])
                 for i in range(self.bones.shape[0])]
        self.lc.set_segments(lines)
        self.fcounter += 1
        self.draw()

    def set_video(self, video):
        self.video = video
        self.fcounter = 0


class ApplicationWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("skeleton display")

        # top-level menu
        file_menu = QMenu('&File', self)
        func_nenu = QMenu('&Function', self)
        help_menu = QMenu('&Help', self)

        # list view
        listModel_1 = QStringListModel()  # models list
        listModel_2 = QStringListModel()  # features list
        listView_1 = QListView()
        listView_1.setModel(listModel_1)
        listView_2 = QListView()
        listView_2.setModel(listModel_2)

        # select action
        listView_2.clicked.connect(self.selectFeature)
        listView_1.clicked.connect(self.selectModel)

        # menu action
        file_menu.addAction(
            '&Import Features', lambda: self.importFeatures(listModel_2), QtCore.Qt.CTRL + QtCore.Qt.Key_F)
        file_menu.addAction(
            '&Import Labels', lambda: self.importLabels(listModel_2), QtCore.Qt.CTRL + QtCore.Qt.Key_L)
        file_menu.addAction(
            '&Import Models', lambda: self.importModels(listModel_1), QtCore.Qt.CTRL + QtCore.Qt.Key_M)
        file_menu.addAction('&Quit', self.quit,
                            QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        func_nenu.addAction('&Predict', lambda: self.predict(listModel_2))
        help_menu.addAction('&About', self.about)

        # add menu
        self.menuBar().addMenu(file_menu)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(func_nenu)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(help_menu)

        # add labels
        label_1 = QLabel(self)
        label_1.setText("Samples")
        label_1.setAutoFillBackground(True)
        label_1.setAlignment(Qt.AlignLeft)
        label_2 = QLabel(self)
        label_2.setText("Models")
        label_2.setAutoFillBackground(True)
        label_2.setAlignment(Qt.AlignLeft)
        label_3 = QLabel(self)
        label_3.setText("Frames")
        label_3.setAutoFillBackground(True)
        label_3.setAlignment(Qt.AlignLeft)

        # window
        self.main_widget = QWidget(self)

        # canvas
        self.canvas = DynamicCanvas(
            self.main_widget, width=5, height=4, dpi=100)

        layout = QGridLayout(self.main_widget)
        layout.addWidget(label_2, 0, 0)
        layout.addWidget(listView_1, 1, 0)
        layout.addWidget(label_1, 0, 1)
        layout.addWidget(listView_2, 1, 1)
        layout.addWidget(label_3, 0, 2)
        layout.addWidget(self.canvas, 1, 2)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.statusBar().showMessage("skeleton display program")

        self.initVariables()

    def initVariables(self):
        self.models = dict()

    def importFeatures(self, listModel):
        fname = QFileDialog.getOpenFileName(
            self, 'open feature file', 'for_display')
        if fname[0] is not None:
            f = h5py.File(fname[0], 'r')
            self.features = np.swapaxes(f['features'][:], 0, 3)
            self.features = np.swapaxes(self.features, 1, 2)
            n_features = self.features.shape[0]

            idx = np.arange(n_features).astype(str)
            self.stringLabeler = StringLabeler({'ID': idx})
            listModel.setStringList(self.stringLabeler.getContent())

    def importLabels(self, listModel):
        fname = QFileDialog.getOpenFileName(
            self, 'open label file', 'for_display')
        if fname[0] is not None:
            f = h5py.File(fname[0], 'r')
            self.te_labels = f['action_labels'][:, 0]

            self.stringLabeler.addItems({'L': self.te_labels})
            listModel.setStringList(self.stringLabeler.getContent())

    def importModels(self, listModel):
        fpath = QFileDialog.getOpenFileName(
            self, 'open model file', 'for_display')
        for fp in fpath[:-1]:
            if fp is not None:
                model = load_model(fp)
                mname = os.path.splitext(os.path.basename(fp))[0]
                self.models[mname] = model

                n_rows = listModel.rowCount()
                if listModel.insertRow(n_rows):
                    i = listModel.index(n_rows)
                    listModel.setData(i, mname)

    def selectFeature(self, modelIndex):
        self.selectedfIndex = modelIndex.row()
        self.selectedFeature = self.features[self.selectedfIndex]
        self.canvas.set_video(self.selectedFeature)

    def selectModel(self, modelIndex):
        self.selectedModel = self.models[modelIndex.data()]

    def predict(self, listModel):
        pr_label = classification_pipline(
            self.selectedModel, self.selectedFeature)

        self.stringLabeler.addSingleItem({'P': pr_label},
                                         self.selectedfIndex)
        listModel.setStringList(self.stringLabeler.getContent())

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
