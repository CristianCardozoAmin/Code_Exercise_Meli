"""
preprocessing.py
last update: 02/03/2025
create by: cristian cardozo amin - cristiancardozo1914@gmail.com
--------------------

Description:

"""

import pandas as pd
import json
import os
import requests
from datetime import datetime
import numpy as np

def normalize_key(row):
    """
    """
    if isinstance(row, list):
        try:
            df_norm = pd.json_normalize(row, sep='_')
            if not df_norm.empty:
                return df_norm.iloc[0].to_dict()
        except Exception:
            return {}
    return {}

def get_columns_nested(df):
    """
    """
    return [
        col for col in df.columns 
        if df[col].dropna().apply(lambda x: isinstance(x, list) and any(isinstance(item, dict) for item in x)).any()
    ]

def explot_columns_nested(df):
    """
    expande los los campos anidados de un .json
    """
    cols_nested = get_columns_nested(df)

    if not cols_nested:
        return df
    
    for col in cols_nested:
        df_norm = df[col].apply(normalize_key)
        df_norm = pd.json_normalize(df_norm)
        df_norm = df_norm.add_prefix(f"{col}_")
        df = pd.concat([df, df_norm], axis=1)

    df.drop(columns=cols_nested, inplace=True)

    return explot_columns_nested(df)


def remove_blanks_columns_name(df:pd.DataFrame) -> pd.DataFrame:
    """
    Carga un dataframe y quita los espacion por _
    
    """
    df.columns = df.columns.str.replace('.', '_')
    return df


def load_dataset(data_source) -> pd.DataFrame:
    """
    Carga un copnjunto de datos, que puede provenir desde diferentes fuentes para generar un dataframe funcional
    
    """
    #Validamos si es un dataframe en caso de que lo sea se copia
    if isinstance(data_source, pd.DataFrame):
        return remove_blanks_columns_name(data_source)
    
    #En caso que se entregue una ruta, validamos que es y cargamos 
    if isinstance(data_source, str):
        ruta, extension = os.path.splitext(data_source)
        extension = extension.lower()

    #validamos la extension y cargamos con la funcion adecuada
    match extension:
        case ".jsonlines":
            with open(data_source, 'r') as file:
                dataframe_raw = [json.loads(line) for line in file]
            dataframe_raw = pd.json_normalize(dataframe_raw)
        case ".json":
            dataframe_raw = pd.read_json(data_source)
            dataframe_raw = pd.json_normalize(dataframe_raw)
        case ".csv":
            dataframe_raw = pd.read_csv(data_source)
        case ".xls" | ".xlsx":
            dataframe_raw = pd.read_excel(data_source)
        case _:
            raise ValueError(f"Extensión de archivo no soportada: {extension}")
    
    return explot_columns_nested(remove_blanks_columns_name(dataframe_raw))