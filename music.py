import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from music_features import convert_mp3_to_wav, print_label

sns.set(color_codes=True)
df = pd.read_csv('dataset.csv')
df1 = df.copy()
df1.drop('filename', axis=1, inplace=True)
le = preprocessing.LabelEncoder()
le.fit(df1['label'])
df1['label'] = le.transform(df1['label'])
df1['label'].unique()
df1['label'].value_counts(normalize=True)
X_train, X_test, y_train, y_test = train_test_split(df1.drop('label', axis=1), df1['label'], test_size=.3,
                                                    random_state=22)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(1, 2, 1), max_iter=10000)
mlp.fit(X_train, y_train)
data = convert_mp3_to_wav("ucl_anthem.mp3")
predictions = mlp.predict(data)
u, c = np.unique(predictions, return_counts=True)
prediction = u[c == c.max()]
print(prediction)
# y_pred_report = np.full(len(y_train), prediction)
#report = classification_report(predictions, y_train);
print(print_label(prediction))
# print(classification_report(y_pred_report, y_train))
