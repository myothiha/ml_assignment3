import pickle

filename = '/root/source_code/model/car_prediction.model'


loaded_model = pickle.load(open(filename, 'rb'))
model = loaded_model['model']
scaler = loaded_model['scaler']
brand_le = loaded_model['brand_le']
default_values = loaded_model['default_values']