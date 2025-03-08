"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import json
import preprocessing
import transformation
import model_processing
from sklearn.model_selection import train_test_split
import os



# You can safely assume that `build_dataset` is correctly implemented
def build_dataset(dataset_path):
    data = preprocessing.load_dataset(dataset_path)
    data = transformation.transform_dataframe(data, conversion_rate=15.0)
    data = model_processing.preprocess_dataframe_model(data)
    data.columns = data.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['condition_used']) 
                                                        , data['condition_used'], test_size=0.3
                                                        , random_state=20250307
                                                        , stratify=data['condition_used'])
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":

    print("Loading dataset...")
    location = os.path.dirname(os.path.abspath(__file__)) 
    dataset_path = '../data/raw/MLA_100k_checked_v3.jsonlines'
    dataset_path = os.path.normpath(os.path.join(location, dataset_path))

    X_train, y_train, X_test, y_test = build_dataset(dataset_path)

    print("training model...")
    model_path = '../data/models/xgb_model_gen.pkl'
    model_path = os.path.normpath(os.path.join(location, model_path))
    modelo_optimizado = model_processing.create_model_and_save(X_train, y_train, ruta_modelo=model_path)

    print("testing model...")
    model_processing.evaluate_model(X_test,y_test,modelo_optimizado)


