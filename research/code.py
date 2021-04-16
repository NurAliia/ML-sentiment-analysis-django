import json # will be needed for saving preprocessing details
import numpy as np # for data manipulation
import pandas as pd # for data manipulation
from sklearn.model_selection import train_test_split # will be used for data split
from sklearn.preprocessing import LabelEncoder # for preprocessing
from sklearn.ensemble import RandomForestClassifier # for training the algorithm
import joblib # for saving algorithm and preprocessing objects

# Read data from xlsx file and fill "nan" data to "0"
df = pd.read_csv('https://raw.githubusercontent.com/NurAliia/ML-sentiment-analysis/dev/data/data.csv').fillna(0);

# X – features, Y – highlighted feature, result
X = df.drop((['Sentiment']), axis=1)
y = df['Sentiment']

# show first rows of data
df.head()

# Split data into training sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# fill missing values
train_mode = dict(X_train.mode().iloc[0])
X_train = X_train.fillna(train_mode)

# Train the Random Forest Classifier model
rf = RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

# save preprocessing objects and RF algorithm
joblib.dump(train_mode, "./train_mode.joblib", compress=True)
joblib.dump(rf, "./random_forest.joblib", compress=True)