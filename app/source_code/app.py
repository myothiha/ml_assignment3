from flask import Flask, render_template, request
from model.load_model import model, scaler, brand_le, default_values
from model.regression_classes import *;

import numpy as np
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # return '<h1>Hello from Flask & Docker</h2>'
    return render_template("index.html", brands = brand_le.classes_, prediction = 0, default_values=default_values)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.form['max_power']:
        max_power = default_values['max_power']
    else:
        max_power = float(request.form['max_power'])

    if not request.form['mileage']:
        mileage = default_values['mileage']
    else:
        mileage = float(request.form['mileage'])

    if not request.form['year']:
        year = default_values['year']
    else:
        year = float(request.form['year'])

    if not request.form['brand']:
        brand = default_values['brand']
    else:
        brand = brand_le.transform([request.form['brand']])

    input_features = np.array([[max_power, mileage, year, int(brand[0])]])
    input_features[:, 0: 3] = scaler.transform(input_features[:, 0: 3])

    prediction = np.exp(model.predict(input_features))
    prediction = format(prediction[0], ".2f")

    return render_template('index.html', prediction=prediction, brands = brand_le.classes_, default_values=default_values)

@app.route('/assignment2', methods=['GET'])
def assignment2():
    load_model = load_model2()

    brand_ohe = load_model['brand_ohe']

    # return '<h1>Hello from Flask & Docker</h2>'
    return render_template("assignment2.html", default_values=default_values, brands = brand_ohe.categories_[0])

@app.route('/a2_predict', methods=['POST'])
def a2_predict():

    load_model = load_model2()

    model = load_model['model']
    scaler = load_model['scaler']
    brand_ohe = load_model['brand_ohe']

    if not request.form['max_power']:
        max_power = default_values['max_power']
    else:
        max_power = float(request.form['max_power'])

    if not request.form['mileage']:
        mileage = default_values['mileage']
    else:
        mileage = float(request.form['mileage'])

    if not request.form['year']:
        year = default_values['year']
    else:
        year = float(request.form['year'])

    # brand = brand_ohe.transform([request.form['brand']])
    encoded_brand = list(brand_ohe.transform([['Skoda']]).toarray()[0])

    input_features = np.array([[max_power, mileage, year] + encoded_brand])
    input_features[:, 0: 3] = scaler.transform(input_features[:, 0: 3])
    input_features = np.insert(input_features, 0, 1, axis=1)

    # input_features = PolynomialFeatures(degree = 3).fit_transform(input_features)
    prediction = np.exp(model.predict(input_features))
    prediction = format(prediction[0], ".2f")

    return render_template('assignment2.html', prediction=prediction, brands = brand_le.classes_, default_values=default_values)


port_number = 80

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)

# if __name__ == "__main__":
#     app.run(debug=True)