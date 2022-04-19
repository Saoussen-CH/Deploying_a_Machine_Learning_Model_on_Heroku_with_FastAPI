import model as md
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import os
from data import process_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('C:/Users/s.chaabnia/Desktop/nd0821-c3-starter-code-master/nd0821-c3-starter-code-master/starter/starter/ml')
from model import train_model, compute_model_metrics, inference
import pytest


def test_train_model(X_train, y_train):
    """
    Check to see if train_model returns a RandomForestClassifier model
    """

    model = train_model(X_train, y_train)
    assert isinstance(model,RandomForestClassifier)



def test_compute_model_metrics(y, preds):
    """
    Check to see if compute_model_metrics returns floats
    """
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference(model, X):
    """
    Check to see if inference returns an array
    """

    preds = inference(model, X)

    assert isinstance(preds,np.ndarray)
