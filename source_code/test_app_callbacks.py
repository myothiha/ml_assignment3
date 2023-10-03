# from contextvars import copy_context
# from dash._callback_context import context_value
# from dash._utils import AttributeDict

import pytest
import numpy as np
# import main
# from pages import model1

from model.load_model import load_meta_data, load_model3

# submit = 1

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

# def test_calculate_y_hardcode_1_plus_2_equal_3():
#     output = model1.calculate_y_hardcode(1,2, submit)
#     assert output == 3

# def test_calculate_y_hardcode_2_plus_2_equal_4():
#     output = model1.calculate_y_hardcode(2,2, submit)
#     assert output == 4

# def test_model_output_shape():
#     output = model1.calculate_model(1,2)
#     assert output.shape == (1,1), f"Expecting the shape to be (1,1) but got {output.shape=}"

# def test_model_coeff_shape():
#     output = model1.get_coeff()
#     assert output.shape == (1,2), f"Expecting the shape to be (1,2) but got {output.shape=}"
