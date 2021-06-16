import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
X_train, X_test, y_train, y_test = train_test_split(df1.drop('label', axis=1), df1['label'], test_size=0.2,
                                                    random_state=22)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("============MLP CLASSIFIER=====================")
mlp = MLPClassifier(hidden_layer_sizes=(10, 20, 10), max_iter=10000, learning_rate='adaptive', solver="sgd",
                    activation="relu")
mlp.fit(X_train, y_train)
prediction_data = convert_mp3_to_wav("Music_Sample/ucl_anthem.mp3")
predictions = mlp.predict(prediction_data)
best_prediction = predictions[np.argsort(predictions)[0]]
best_predictions = np.argsort(predictions)[0:len(y_test)]
prediction_report = []
for i in best_predictions:
    prediction_report.append(predictions[i])
print(print_label(best_prediction))
print(classification_report(y_test, prediction_report))


print("============DECISION TREE=====================")
clf = DecisionTreeClassifier(criterion="gini")
clf.fit(X_train, y_train)
predictions2 = clf.predict(prediction_data)
best_prediction2 = predictions2[np.argsort(predictions2)[0]]
best_predictions2 = np.argsort(predictions2)[0:len(y_test)]
prediction_report2 = []
for i in best_predictions2:
    prediction_report2.append(predictions2[i])
print(classification_report(y_test, prediction_report2))