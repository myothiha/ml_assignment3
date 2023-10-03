import pytest
import numpy as np

from model.load_model import load_meta_data, load_model3

def test_model_input():
    scaler, brand_ohe, default_values, classes = load_meta_data()

    max_power = default_values['max_power']
    mileage = default_values['mileage']
    year = default_values['year']
    brand = "Skoda"
    encoded_brand = list(brand_ohe.transform([['Skoda']]).toarray()[0])

    input_features = np.array([[max_power, mileage, year] + encoded_brand])
    input_features[:, 0: 3] = scaler.transform(input_features[:, 0: 3])
    input_features = np.insert(input_features, 0, 1, axis=1)
    
    assert input_features.shape == (1, 35)

def test_model_output():
    scaler, brand_ohe, default_values, classes = load_meta_data()

    max_power = default_values['max_power']
    mileage = default_values['mileage']
    year = default_values['year']
    brand = "Skoda"
    encoded_brand = list(brand_ohe.transform([['Skoda']]).toarray()[0])

    input_features = np.array([[max_power, mileage, year] + encoded_brand])
    input_features[:, 0: 3] = scaler.transform(input_features[:, 0: 3])
    input_features = np.insert(input_features, 0, 1, axis=1)

    model = load_model3()
    predicted_class = model.predict(input_features)
    
    assert predicted_class.shape == (1, )