# Import necessary packages
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5.QtWidgets import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from music_features import convert_mp3_to_wav, print_label
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')
sns.set(color_codes=True)


class predict_music_genre_window(QDialog):
    def __init__(self):
        super(predict_music_genre_window, self).__init__()
        self.setWindowTitle("MUSIC GENRE PREDICTION")
        self.setFixedWidth(800)
        self.setFixedHeight(1000)
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        self.formGroupBox = QGroupBox("PREDICT MUSIC GENRE")
        fileUploadBtn = QPushButton("Select mp3 File")
        self.result = QPlainTextEdit()
        self.result.setFixedWidth(800)
        self.result.setFixedHeight(600)
        self.result.setDisabled(True)
        self.result.setStyleSheet("color: white;  background-color: black; font-size:12pt")
        self.display_results()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(fileUploadBtn)
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(self.canvas)
        self.setLayout(mainLayout)
        fileUploadBtn.clicked.connect(self.open)
        self.show()

    def predict_music_genre(self, path):
        # Read in dataset
        music_data = pd.read_csv('dataset.csv')

        #   Check df
        self.result.setPlainText('Dataset Shape: ' + str(music_data.shape))
        self.result.appendPlainText(music_data.head().to_string())
        self.result.appendPlainText(str(music_data.describe))
        self.result.appendPlainText('NA Count: ' + str(music_data.isna().sum()))

        # Create new df
        new_music_data = music_data.copy()
        new_music_data.drop('filename', axis=1, inplace=True)
        self.figure.clear()
        # Histogram to check for distribution
        plt.hist(new_music_data['label'], bins=30)
        plt.title('Target Distribution')
        plt.plot()
        self.canvas.draw()

        # Label encode target
        le = preprocessing.LabelEncoder()
        new_music_data['label'] = le.fit_transform(new_music_data['label'])
        self.result.appendPlainText(str(new_music_data['label'].unique()))
        new_music_data['label'].value_counts(normalize=True)

        # Check balance of target
        self.result.appendPlainText(str(new_music_data.groupby('label').count()))

        # Create train/test data
        X = new_music_data.drop('label', axis=1)
        y = new_music_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Standardize features
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Create network
        mlp = MLPClassifier(hidden_layer_sizes=(10, 20, 10), max_iter=10000, learning_rate='adaptive', solver="sgd",
                            activation="relu")

        # Test network
        self.result.appendPlainText("============PREDICT TEST SPLIT WITH MLP CLASSIFIER=====================")
        mlp.fit(X_train, y_train)
        x_predictions = mlp.predict(X_test)
        self.result.setPlainText(classification_report(y_test, x_predictions))

        # Convert mp3 files to wav file type
        features = convert_mp3_to_wav(path)

        # Predict unseen data
        prediction_data = features.iloc[150:]
        self.result.appendPlainText("============PREDICT UPLOADED MP3 FILE WITH MLP CLASSIFIER=====================")
        predictions = mlp.predict(prediction_data)

        # Locate best predictions & test predictions
        best_prediction = predictions[np.argsort(predictions, axis=None)[0]]
        self.result.appendPlainText(print_label(best_prediction))

        # Predict using a Decision Tree Model
        self.result.appendPlainText("============PREDICT UPLOADED MPS WITH SKLEARN DECISION TREE=====================")
        clf = DecisionTreeClassifier(criterion="gini")
        clf.fit(X_train, y_train)
        predictions2 = clf.predict(X_test)
        self.result.appendPlainText(classification_report(y_test, predictions2))

    def open(self):
        path = QFileDialog.getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')
        if path != ('', ''):
            self.predict_music_genre(path[0])

    def display_results(self):
        layout = QFormLayout()
        layout.addRow(QLabel("Prediction Results"), self.result)
        self.formGroupBox.setLayout(layout)
