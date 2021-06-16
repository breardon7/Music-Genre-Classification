import warnings
import pandas as pd
from PyQt5.QtWidgets import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


class predict_music_genre_window(QDialog):
    def __init__(self):
        super(predict_music_genre_window, self).__init__()
        self.setWindowTitle("DATS6103 GROUP PROJECT - TEAM 5 (SOCCER MATCH PREDICTION)")
        self.setFixedWidth(800)
        self.setFixedHeight(1000)
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        self.formGroupBox = QGroupBox("PREDICT SOCCER MATCH")
        fileUploadBtn = QPushButton("Select mp3 File")

        self.result = QPlainTextEdit()
        self.result.setFixedWidth(500)
        self.result.setFixedHeight(200)
        self.result.setDisabled(True)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(fileUploadBtn)
        mainLayout.addWidget(self.canvas)
        self.setLayout(mainLayout)

    def open(self):
        path = QFileDialog.getOpenFileName(self, 'Open a file', '',
                                           'All Files (*.*)')
        if path != ('', ''):
            print("File path : " + path[0])

    def analyzeMatch(self):
        self.figure.clear()
        self.canvas.draw()
        self.result.setPlainText()
        self.result.appendPlainText()
