# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pickle
import pandas as pd

# Add the necessary imports for the starter code.
import os
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, compute_slice_metrics

# Add code to load in the data.
filename = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'census_clean_data.csv'))
data = pd.read_csv(filename)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

filename = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'model', "OneHotEnoder.pkl"))
with open(filename, 'wb') as file:
    pickle.dump(encoder, file)

filename = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'model', "LabelBinarizer.pkl"))
with open(filename, 'wb') as file:
    pickle.dump(lb, file)
    
# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.

model = train_model(X_train, y_train)
filename = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'model', "model.pkl"))
with open(filename, 'wb') as file:
    pickle.dump(model, file)
    
    
# Test model and show metrics
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print('Precision:', precision)

print('Recall:', recall)

print('F-Beta Score:', fbeta)