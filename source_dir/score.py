import json

import joblib
import pandas as pd

from azureml.core import Model

min_max_values = {
    "age": (40.0, 95.0),
    "creatinine_phosphokinase": (23.0, 7861.0),
    "ejection_fraction": (14.0, 80.0),
    "platelets": (25100.0, 850000.0),
    "serum_creatinine": (0.5, 9.4),
    "serum_sodium": (113.0, 148.0),
    "time": (4.0, 285.0)
}

dummy_sizes = {
    'anaemia' : 2, 
    'diabetes': 2, 
    'high_blood_pressure' : 2, 
    'sex' : 2, 
    'smoking' : 2
}



def init():
    global model
    model_path = Model.get_model_path("HearFailurePrediction")
    model = joblib.load(model_path)


def preprocess(x_df):
    normalized_column_names = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium','time']

    data_types = {column_name:"float32" for column_name in normalized_column_names}

    min_values = data[normalized_column_names].min(axis=0)
    max_values = data[normalized_column_names].max(axis=0)

    for column_name in normalized_column_names:
        m0, m1 = min_max_values[column_name]
        x_df[column_name] = x_df[column_name].apply(lambda x : (x - m0)/(m1 - m0))

    category_column_names = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

    for column_name in category_column_names:
        values = x_df[column_name].to_numpy(dtype='int')
        tmp = np.zeros((len(values), dummy_sizes[column_name]), dtype='int')
        tmp[:, values] = 1
        tmp = pd.DataFrame(tmp, columns=[f"{column_name[:3]}_{i}" for i in range(dummy_sizes[column_name])])
        x_df.drop(column_name, inplace=True, axis=1)
        x_df = x_df.join(tmp)

    return x_df


def run(data):
    try:
        test = json.loads(data)
        x_df = pd.DataFrame(test['data'])
        x_df = preprocess(x_df)
        prediction = model.predict(s_df)
        result = json.dumps({"result" : prediction.tolist()})
    except Exception as e:
        result = json.dumps({"Error" : str(e)})

    return result
