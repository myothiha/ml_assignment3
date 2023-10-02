import pickle
import mlflow
import os

filename = '/root/source_code/model/car_prediction.model'


loaded_model = pickle.load(open(filename, 'rb'))
model = loaded_model['model']
scaler = loaded_model['scaler']
brand_le = loaded_model['brand_le']
default_values = loaded_model['default_values']

def load_meta_data():
    filename = '/root/source_code/model/a3_meta_data.dump'
    meta = pickle.load(open(filename, 'rb'))

    scaler = meta['scaler']
    brand_ohe = meta['brand_ohe']
    default_values = meta['default_values']
    classes = meta['classes']
    
    return (scaler, brand_ohe, default_values, classes)

def load_model3():

    mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
    os.environ["LOGNAME"] = "myo"
    mlflow.set_experiment(experiment_name="st123783-myo")

    # Load model from the model registry.
    model_name = "st123783-a3-model"
    model_version = 1
    stage = "Staging"

    # load a specific model version
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    # load the latest version of a model in that stage.
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

    return model