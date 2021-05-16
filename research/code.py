import pandas as pd # for data manipulation
from sklearn.model_selection import train_test_split # will be used for data split
from sklearn.ensemble import RandomForestClassifier # for training the algorithm
import joblib # for saving algorithm and preprocessing objects

# Read data from xlsx file and fill "nan" data to "0"
df = pd.read_excel('https://raw.githubusercontent.com/NurAliia/ML-sentiment-analysis/dev/data/data2020.xlsx').fillna(0);

# X – features, Y – highlighted feature, result
X = df.drop((['Текст', 'Тональность']), axis=1)
y = df['Тональность']

# show first rows of data
df.head()

# Split data into training sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train the Random Forest Classifier model
rf = RandomForestClassifier(min_samples_split=3, max_features='log2')
rf.fit(X_train, y_train)

# save preprocessing objects and RF algorithm
joblib.dump(X_train, "./train_mode.joblib", compress=True)
joblib.dump(rf, "./random_forest.joblib", compress=True)
