import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


dataDict = pickle.load(open('./images.pickle', 'rb'))

data = np.asarray(dataDict['data'])
labels = np.asarray(dataDict['labels'])

# splits data and corresponding labels into training(0.8) & testing(0.2) sets 
# Training set > x_train: input features,  y_train: labels
# Test set > x_test,  y_test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Training method
model = RandomForestClassifier()

# Model Training
model.fit(x_train, y_train)

# Storing the predicted labels (from x_test data) in y_predict
y_predict = model.predict(x_test)

# Evaluation: Compare y_predict and y_test
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)