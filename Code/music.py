# Import necessary packages
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from music_features import convert_mp3_to_wav, print_label
warnings.filterwarnings('ignore')

sns.set(color_codes=True)

# Read in dataset
music_data = pd.read_csv('dataset.csv')

# Check df
print('df shape', music_data.shape)
print(music_data.head())
print(music_data.describe)
print('NA count', music_data.isna().sum())

# Create new df
new_music_data = music_data.copy()
new_music_data.drop('filename', axis=1, inplace=True)

# Histogram to check for distribution
plt.hist(new_music_data['label'], bins = 30)
plt.title('Target Distribution')
plt.plot()

# Label encode target
le = preprocessing.LabelEncoder()
new_music_data['label'] = le.fit_transform(new_music_data['label'])
print(new_music_data['label'].unique())
new_music_data['label'].value_counts(normalize=True)

# Check balance of target
print(new_music_data.groupby('label').count())

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
print("============PREDICT TEST SPLIT WITH MLP CLASSIFIER=====================")
mlp.fit(X_train, y_train)
x_predictions = mlp.predict(X_test)
print(classification_report(y_test, x_predictions))
print(confusion_matrix(y_test, x_predictions))

# Convert mp3 files to wav file type
features = convert_mp3_to_wav("Music_Sample/ucl_anthem.mp3")

# Predict unseen data
prediction_data = features.iloc[150:]
print("============PREDICT UPLOADED MP3 FILE WITH MLP CLASSIFIER=====================")
predictions = mlp.predict(prediction_data)

# Locate best predictions & test predictions
best_prediction = predictions[np.argsort(predictions, axis=None)[0]]
best_predictions = np.argsort(predictions)[0:len(y_test)]
prediction_report = []
for i in best_predictions:
    prediction_report.append(predictions[i])
print(print_label(best_prediction))
print(classification_report(y_test, prediction_report))
print(confusion_matrix(y_test, prediction_report))

# Predict using a Decision Tree Model
print("============PREDICT UPLOADED MPS WITH SKLEARN DECISION TREE=====================")
clf = DecisionTreeClassifier(criterion="gini")
clf.fit(X_train, y_train)
predictions2 = clf.predict(prediction_data)
best_prediction2 = predictions2[np.argsort(predictions2, axis=None)[0]]
best_predictions2 = np.argsort(predictions2)[0:len(y_test)]
prediction_report2 = []
for i in best_predictions2:
    prediction_report2.append(predictions2[i])
print(classification_report(y_test, prediction_report2))
print(confusion_matrix(y_test, prediction_report2))