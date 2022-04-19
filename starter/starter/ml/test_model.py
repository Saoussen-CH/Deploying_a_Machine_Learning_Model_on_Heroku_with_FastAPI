import model as md
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import os
from .data import process_data
import numpy as np
import sys
sys.path.append('C:/Users/s.chaabnia/Desktop/nd0821-c3-starter-code-master/nd0821-c3-starter-code-master/starter/starter/ml')
from .model import train_model, compute_model_metrics, inference
import pytest



@pytest.fixture
def data():
    """ Retrieve Cleaned Dataset """
    train_file = "starter/data/census_clean_data.csv"
    df = pd.read_csv(train_file)

    return df

def test_train_model(data):
    """Tests whether the model its work and built a file"""
    train, test =  train_test_split(data, test_size=.20)
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
    model = train_model(X_train, y_train)
    filepath = "starter/model/model.pkl"
    assert os.path.exists(filepath)


@pytest.fixture
def split_data(data):
    """Split data in train test"""
    train, test = train_test_split(data, test_size=.20)
    return (train, test)

def test_inference(split_data):
    train, test = split_data
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
    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb      
    )
    path_model = "starter/model/model.pkl"
    model = pickle.load(open(path_model, 'rb'))

    y_train_pred = inference(model, X_train)
    assert len(y_train_pred) == X_train.shape[0]
    assert len(y_train_pred) > 0

    y_test_pred = inference(model, X_test)
    assert len(y_test_pred) == X_test.shape[0]
    assert len(y_test_pred) > 0


def test_compute_model_metrics(split_data):
    train, test = split_data
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
    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb      
    )
    path_model = "starter/model/model.pkl"
    model = pickle.load(open(path_model, 'rb'))

    y_train_pred = inference(model, X_train)   
    y_test_pred = inference(model, X_test)

    precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, y_train_pred)
    precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, y_test_pred)

    assert isinstance(precision_train, float)
    assert isinstance(precision_test, float)
    assert isinstance(recall_train, float)
    assert isinstance(recall_test, float)
    assert isinstance(fbeta_train, float)
    assert isinstance(fbeta_test, float)

    assert (precision_train<=1) & (precision_train>=0)
    assert (precision_test<=1) & (precision_test>=0)
    assert (recall_train<=1) & (recall_train>=0)
    assert (recall_test<=1) & (recall_test>=0)
    assert (fbeta_train<=1) & (fbeta_train>=0)
    assert (fbeta_test<=1) & (fbeta_test>=0)
