import os
import argparse

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from azureml.core import Workspace
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory



def clean_data(data):
    normalized_column_names = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium','time']

    x_df = data.to_pandas_dataframe().dropna()

    min_values = x_df[normalized_column_names].min(axis=0)
    max_values = x_df[normalized_column_names].max(axis=0)

    for column_name in normalized_column_names:
        m0 = min_values[column_name]
        m1 = max_values[column_name]
        # print(m0, m1)

        x_df[column_name] = x_df[column_name].apply(lambda x : (x - m0)/(m1 - m0))

    category_column_names = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

    for column_name in category_column_names:
        tmp = pd.get_dummies(x_df[column_name], prefix=column_name[:3])
        x_df.drop(column_name, inplace=True, axis=1)
        x_df = x_df.join(tmp)

    y_df = x_df.pop('DEATH_EVENT')

    return x_df, y_df

# Set up workspace and its resource
ws = Workspace.from_config()
datastore = ws.get_default_datastore()

# Create TabularDataset using TabularDatasetFactory
filename = datastore.path("heart_failure_clinical_records_dataset.csv")
ds = TabularDatasetFactory.from_delimited_files(filename)

x, y = clean_data(ds)

# Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

run = Run.get_context(allow_offline=True, used_for_context_manager=False)


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))

    os.makedirs(os.path.join(".", "outputs"), exist_ok=True)
    with open(os.path.join(".", "outputs", f"model_{args.C:.4f}_{args.max_iter:d}.joblib"), "wb") as f:
        joblib.dump(model, f)

if __name__ == '__main__':
    main()

