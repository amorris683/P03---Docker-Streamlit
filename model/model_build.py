import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

# load CHLA data
df = pd.read_csv("CHLA_clean_data_until_2023.csv")

# typecast columns to their correct data type
df['MRN'] = df['MRN'].astype(str)
df['APPT_DATE'] = pd.to_datetime(df['APPT_DATE'])
df['BOOK_DATE'] = pd.to_datetime(df['BOOK_DATE'])
df['SCHEDULE_ID'] = df['SCHEDULE_ID'].astype(str)
df['APPT_ID'] = df['APPT_ID'].astype(str)
df['WEEK_OF_MONTH'] = df['WEEK_OF_MONTH'].astype(str)

# drop duplicates
df = df.drop_duplicates()

df = df[['AGE', 'CLINIC', 'TOTAL_NUMBER_OF_CANCELLATIONS', 'LEAD_TIME', 'TOTAL_NUMBER_OF_RESCHEDULED', 'TOTAL_NUMBER_OF_NOSHOW',
         'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT', 'HOUR_OF_DAY', 'NUM_OF_MONTH', 'IS_NOSHOW']]

# import lable encoder
from sklearn.preprocessing import LabelEncoder

# Create a copy of the DataFrame to avoid changing the original one
df_encoded = df.copy()

# Create a LabelEncoder object
le = LabelEncoder()

# List of columns to encode
object_cols = ['CLINIC', 'IS_NOSHOW']

# Apply the encoder to each column
for col in object_cols:
    df_encoded[col] = le.fit_transform(df[col])

import pickle

# Serialize the LabelEncoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

print("LabelEncoder has been serialized as label_encoder.pkl")

from sklearn.model_selection import train_test_split

# Features and label

X = df_encoded.drop('IS_NOSHOW', axis=1)  # All columns except 'IS_NOSHOW'
y = df_encoded['IS_NOSHOW']  # Only the 'IS_NOSHOW' column

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% of the data will be used for testing

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

classifiers = {
    'LogisticRegression': (LogisticRegression(), {'C': np.logspace(-3, 3, 7), 'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs']}),  # 'l1' penalty is only supported with 'liblinear' solver
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30]}),
    'SVC': (SVC(), {'C': np.logspace(-3, 3, 7), 'kernel': ['linear', 'rbf']})
}

best_model = None
best_score = 0
best_clf_name = ""

# Loop through classifiers
for clf_name, (clf, params) in classifiers.items():
    grid_search = GridSearchCV(clf, params, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    print(f"{clf_name} best parameters: {grid_search.best_params_}, best score: {grid_search.best_score_}")

    # Update the best model if this model is better
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_clf_name = clf_name

# Serialize the best model
with open(f'best_model_{best_clf_name}.pkl', 'wb') as f:
    pickle.dump(best_model, f)