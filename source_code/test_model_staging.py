"""
The idea of this file is to test the model at Staging.
If it passes the test, we will automtically move the Model to Production.

This file will test the model in Staging.
If the model in Staging is tested, on the production level, 
"""
from utils import load_mlflow
import numpy as np
import pandas as pd
import pytest
# I don't need to set mlflow.set_tracking_uri()
# because I set it in the environment of this container during compose up.
# With this, people who has my image won't know the link to the mlflow server.
stage = "Staging"
def test_load_model():
    model = load_mlflow(stage=stage)
    assert model

@pytest.mark.depends(on=['test_load_model'])
def test_model_input():
    model = load_mlflow(stage=stage)
    X = np.array([1,2]).reshape(-1,2)
    X = pd.DataFrame(X, columns=['x1', 'x2']) 
    pred = model.predict(X) # type:ignore
    assert pred

@pytest.mark.depends(on=['test_model_input'])
def test_model_output():
    model = load_mlflow(stage=stage)
    X = np.array([1,2]).reshape(-1,2)
    X = pd.DataFrame(X, columns=['x1', 'x2']) 
    pred = model.predict(X) # type:ignore
    assert pred.shape == (1,1), f"{pred.shape=}"

@pytest.mark.depends(on=['test_load_model'])
def test_model_coeff():
    model = load_mlflow(stage=stage)
    assert model.coef_.shape == (1,2), f"{model.coef_.shape=}" # type:ignore