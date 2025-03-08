from fastapi import FastAPI, UploadFile, File
import pandas as pd
from src import preprocessing, transformation, model_processing
import os
import joblib
from io import BytesIO 

app = FastAPI()

#modelo entrenado y testiado
location = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(location, '../data/models/xgb_model_gen.pkl'))
modelo_optimizado = joblib.load(model_path)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer archivo subido
    contents = await file.read()
    input_df = pd.read_json(BytesIO(contents), lines=True)

    #preporcesamos el dataframe
    data = preprocessing.load_dataset(input_df)
    data = transformation.transform_dataframe(data, conversion_rate=15.0)
    data = model_processing.preprocess_dataframe_model(data)
    data.columns = data.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

    #generammos predicciones
    predictions = modelo_optimizado.predict(data)

    #predicciones en formato JSON
    return {"predictions": predictions.tolist()}