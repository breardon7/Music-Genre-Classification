# ------------------------------------------------------------------------
# Parameter Grid test
# ------------------------------------------------------------------------

# Import Packages

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from music_features import convert_mp3_to_wav, print_label
from sklearn.model_selection import GridSearchCV


# Read in dataset
music_data = pd.read_csv('dataset.csv')

#   Check df
print((music_data.shape))
print(music_data.head())
print(music_data.describe)
#print('NA Count: ' + music_data.isna().sum())

# Create new df
new_music_data = music_data.copy()
new_music_data.drop('filename', axis=1, inplace=True)

# Histogram to check for distribution
plt.hist(new_music_data['label'], bins=30)
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
mlp = MLPClassifier(max_iter=1000000)

# Hyper-parameter space
'''parameter_space = {
    'hidden_layer_sizes': [(60,100,60), (100,100,100), (50,100,50)],
    'activation': ['identity', 'relu', 'logistic', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.00001, 0.000001],
    'learning_rate': ['constant','adaptive', 'invscaling'],
}'''

parameter_space = {
    'hidden_layer_sizes': [(60,100,60), (60,60,60), (70,20,70), (50,50,50)],
    'activation': ['tanh'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['invscaling'],
}

# Run Gridsearch
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)


# Test network
print("============PREDICT TEST SPLIT WITH MLP CLASSIFIER=====================")
mlp.fit(X_train, y_train)
x_predictions = mlp.predict(X_test)
print(classification_report(y_test, x_predictions))

print("============PREDICT TEST SPLIT WITH Best_Params MLP CLASSIFIER=====================")
clf.fit(X_train, y_train)
x_predictions = clf.predict(X_test)
print(classification_report(y_test, x_predictions))

# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# Convert mp3 files to wav file type
path = 'C:\\Users\\brear\\OneDrive\\Desktop\\Grad School\\Music-Genre-Classification\\Code\\Music_Sample\\changes.mp3'
features = convert_mp3_to_wav(path)

# Predict unseen data
prediction_data = features.iloc[150:]
print("============PREDICT UPLOADED MP3 FILE WITH MLP CLASSIFIER=====================")
predictions = mlp.predict(prediction_data)

# Locate best predictions & test predictions
best_prediction = predictions[np.argsort(predictions, axis=None)[0]]
print(print_label(best_prediction))

# Predict using a Decision Tree Model
print("============PREDICT UPLOADED MPS WITH SKLEARN DECISION TREE=====================")
clf = DecisionTreeClassifier(criterion="gini")
clf.fit(X_train, y_train)
predictions2 = clf.predict(X_test)
print(classification_report(y_test, predictions2))

# Best Params
'''Best parameters found:
 {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50), 'learning_rate': 'invscaling', 'solver': 'adam'}'''
'''Best parameters found:
 {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (60, 100, 60), 'learning_rate': 'invscaling', 'solver': 'adam'}'''