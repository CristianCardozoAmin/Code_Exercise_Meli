import pandas as pd
import numpy as np
import requests
from datetime import datetime

########################################
# Funciones Genéricas Optimized
########################################

def transform_date_column(df, col, prefix):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df[f"{prefix}_year"] = df[col].dt.year
    df[f"{prefix}_month"] = df[col].dt.month
    df[f"{prefix}_day"] = df[col].dt.day
    df[f"{prefix}_weekday"] = df[col].dt.weekday
    df[f"{prefix}_iso_week"] = df[col].dt.isocalendar().week
    df[f"{prefix}_hour"] = df[col].dt.hour
    df[f"{prefix}_minute"] = df[col].dt.minute
    df[f"{prefix}_second"] = df[col].dt.second
    df.drop(columns=[col], inplace=True)
    return df

def transform_date_column2(df, col, prefix):
    df[col] = pd.to_datetime(df[col], unit='ms')
    df[f"{prefix}_year"] = df[col].dt.year
    df[f"{prefix}_month"] = df[col].dt.month
    df[f"{prefix}_day"] = df[col].dt.day
    df[f"{prefix}_weekday"] = df[col].dt.weekday
    df[f"{prefix}_iso_week"] = df[col].dt.isocalendar().week
    df[f"{prefix}_hour"] = df[col].dt.hour
    df[f"{prefix}_minute"] = df[col].dt.minute
    df[f"{prefix}_second"] = df[col].dt.second
    df.drop(columns=[col], inplace=True)
    return df


def normalize_numeric_column(df, col):
    threshold = df[col].quantile(0.95)
    df[f"{col}_normalized"] = np.where(df[col] <= threshold, df[col] / threshold, 1)
    df.drop(columns=[col], inplace=True)
    return df

def create_boolean_from_notnull(df, col, new_col=None):
    if new_col is None:
        new_col = "has_" + col
    df[new_col] = df[col].notnull()
    df.drop(columns=[col], inplace=True)
    return df

def transform_top_categories(df, col, new_col, top_n=5, others_label='otros'):
    top_categories = df[col].value_counts().nlargest(top_n).index
    df[new_col] = np.where(df[col].isin(top_categories), df[col], others_label)
    df.drop(columns=[col], inplace=True)
    return df

def transform_picture_size_column(df, col, new_prefix):
    regex = r'(?P<width>\d+(?:\.\d+)?)\s*[xX]\s*(?P<height>\d+(?:\.\d+)?)'
    extracted = df[col].astype(str).str.extract(regex)
    df[f"{new_prefix}_width"] = pd.to_numeric(extracted['width'], errors='coerce')
    df[f"{new_prefix}_height"] = pd.to_numeric(extracted['height'], errors='coerce')
    df.drop(columns=[col], inplace=True)
    return df

def transform_price_column(df, col, new_converted_col, new_standardized_col, conversion_rate, currency_col='currency_id'):
    if currency_col in df.columns:
        currencies = df[currency_col].fillna('ARS').astype(str).str.upper()
        multiplier = np.where(currencies.isin(['USD', 'US']), conversion_rate, 1)
    else:
        multiplier = 1
    df[new_converted_col] = df[col] * multiplier
    mean_val = df[new_converted_col].mean()
    std_val = df[new_converted_col].std()
    df[new_standardized_col] = (df[new_converted_col] - mean_val) / std_val
    df.drop(columns=[col], inplace=True)
    return df

########################################
# Funciones Específicas Optimized
########################################

def transform_warranty(df):
    keywords = ['mes', 'año', 'dia', 'día']
    pattern = '|'.join(keywords)
    df['warranty_flag'] = df['warranty'].astype(str).str.lower().str.contains(pattern)
    df.drop(columns=['warranty'], inplace=True)
    return df

def transform_condition(df):
    df['condition_used'] = df['condition'].astype(str).str.lower() == 'used'
    df.drop(columns=['condition'], inplace=True)
    return df

def transform_listing_type_id(df):
    mapping = {'free': 1, 'bronce': 2}
    df['listing_type_id_ordinal'] = df['listing_type_id'].astype(str).str.lower().map(mapping).fillna(3).astype(int)
    df.drop(columns=['listing_type_id'], inplace=True)
    return df

def transform_tags(df):
    df['tags'] = df['tags'].astype(str).str.strip("[]").str.replace("'", "").str.strip()
    df['tags'] = df['tags'].apply(lambda x: ','.join(tag.strip() for tag in x.split(',')))
    tags_dummies = df['tags'].str.get_dummies(sep=',').add_prefix('tag_')
    df = pd.concat([df, tags_dummies], axis=1)
    df.drop(columns=['tags'], inplace=True)
    return df

def transform_seller_address_state(df):
    df['is_capital_state'] = df['seller_address_state_name'].fillna('').str.strip().str.lower().isin(['capital federal', 'buenos aires'])
    df.drop(columns=['seller_address_state_name'], inplace=True)
    return df

def transform_payment_methods_type(df):
    df['non_mercado_pago_payment_methods_type_flag'] = df['non_mercado_pago_payment_methods_type'].astype(str).str.upper() == 'G'
    df.drop(columns=['non_mercado_pago_payment_methods_type'], inplace=True)
    return df

def transform_seller_address_city_id(df):
    unique_ids = df['seller_address_city_id'].unique()
    cache = {}
    for cid in unique_ids:
        url = f"https://api.mercadolibre.com/classified_locations/cities/{cid}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                location = data.get('geo_information', {}).get('location', {})
                cache[cid] = (location.get('latitude', np.nan), location.get('longitude', np.nan))
            else:
                cache[cid] = (np.nan, np.nan)
        except:
            cache[cid] = (np.nan, np.nan)
    df['seller_city_latitude'] = df['seller_address_city_id'].map(lambda x: cache.get(x, (np.nan, np.nan))[0])
    df['seller_city_longitude'] = df['seller_address_city_id'].map(lambda x: cache.get(x, (np.nan, np.nan))[1])
    df.drop(columns=['seller_address_city_id'], inplace=True)
    return df

def transform_status(df):
    df['status_active'] = df['status'].astype(str).str.lower() == 'active'
    df.drop(columns=['status'], inplace=True)
    return df

def transform_shipping_mode(df):
    if 'shipping_mode' in df.columns:
        shipping_mode_dummies = pd.get_dummies(df['shipping_mode'], prefix='shipping_mode')
        df = pd.concat([df, shipping_mode_dummies], axis=1)
        df.drop(columns=['shipping_mode'], inplace=True)
    else:
        print("Columna 'shipping_mode' no encontrada; creando columnas dummy vacías.")
        df['shipping_mode_desconocido'] = 1 
    return df

########################################
# Función Principal Optimized
########################################

def transform_dataframe(df, conversion_rate=15.0):

    print(df.columns)
    # Primer bloque de transformaciones
    print('Primer bloque de transformaciones')
    df = transform_warranty(df)
    df = transform_condition(df)
    df = transform_price_column(df, 'base_price', 'base_price_converted', 'base_price_standardized', conversion_rate)
    df = transform_listing_type_id(df)
    df = transform_price_column(df, 'price', 'price_converted', 'price_standardized', conversion_rate)
    df = transform_tags(df)
    df = create_boolean_from_notnull(df, 'parent_item_id', 'has_parent_item')
    df = transform_date_column(df, 'last_updated', 'last_updated')
    
    # Se combinan columnas a eliminar en un solo paso (algunas ya se eliminaron en cada función)
    print('Drop tables')
    cols_to_drop = ['substatus', 'deal_ids', 'site_id', 'buying_mode', 'listing_source', 
                    'coverage_areas', 'category_id', 'descriptions', 'international_delivery_mode',
                    'id', 'official_store_id', 'differential_pricing', 'original_price', 'title', 
                    'secure_thumbnail', 'catalog_product_id', 'subtitle', 'permalink', 'pictures_secure_url','pictures_url', 'pictures_id',
                    'thumbnail','seller_address_country_name','seller_address_country_id','seller_address_state_id','seller_address_city_name',
                    'shipping_methods','shipping_tags','variations_seller_custom_field','sub_status'
                    ,'seller_id','shipping_free_methods_rule_value','shipping_dimensions']
    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    print('Segundo bloque de transformaciones')
    # Segundo bloque de transformaciones
    df = transform_date_column(df, 'date_created', 'date_created')
    df = transform_date_column2(df, 'stop_time', 'stop_time')
    df = transform_date_column2(df, 'start_time', 'start_time')
    df = transform_status(df)
    df = create_boolean_from_notnull(df, 'video_id', 'has_video')
    df = normalize_numeric_column(df, 'initial_quantity')
    df = normalize_numeric_column(df, 'sold_quantity')
    df = normalize_numeric_column(df, 'available_quantity')
    df = transform_shipping_mode(df)
    df = transform_top_categories(df, 'non_mercado_pago_payment_methods_description',
                                  'non_mercado_pago_payment_methods_description_transformed', top_n=5)
    df = transform_top_categories(df, 'non_mercado_pago_payment_methods_id',
                                  'non_mercado_pago_payment_methods_id_transformed', top_n=5)
    df = transform_payment_methods_type(df)
    df = create_boolean_from_notnull(df, 'variations_picture_ids', 'has_variations_picture_ids')
    df = create_boolean_from_notnull(df, 'variations_sold_quantity', 'has_variations_sold_quantity')
    df = create_boolean_from_notnull(df, 'variations_id', 'has_variations_id')
    df = create_boolean_from_notnull(df, 'variations_price', 'has_variations_price')
    df = create_boolean_from_notnull(df, 'attributes_value_id', 'has_attributes_value_id')
    df = transform_top_categories(df, 'attributes_attribute_group_id',
                                  'attributes_attribute_group_id_transformed', top_n=2)
    df = transform_top_categories(df, 'attributes_name', 'attributes_name_transformed', top_n=2)
    df = create_boolean_from_notnull(df, 'attributes_value_name', 'has_attributes_value_name')
    df = transform_top_categories(df, 'attributes_attribute_group_name',
                                  'attributes_attribute_group_name_transformed', top_n=2)
    df = transform_top_categories(df, 'attributes_id', 'attributes_id_transformed', top_n=2)
    df = transform_picture_size_column(df, 'pictures_size', 'picture')
    df = transform_picture_size_column(df, 'pictures_max_size', 'picture_max')
    df = create_boolean_from_notnull(df, 'shipping_free_methods_id', 'has_shipping_free_methods_id')
    df = create_boolean_from_notnull(df, 'shipping_free_methods_rule_free_mode', 'has_shipping_free_methods_rule_free_mode')
    df = create_boolean_from_notnull(df, 'variations_attribute_combinations_value_id', 'has_variations_attribute_combinations_value_id')
    df = create_boolean_from_notnull(df, 'variations_attribute_combinations_name', 'has_variations_attribute_combinations_name')
    df = create_boolean_from_notnull(df, 'variations_attribute_combinations_value_name', 'has_variations_attribute_combinations_value_name')
    df = create_boolean_from_notnull(df, 'variations_attribute_combinations_id', 'has_variations_attribute_combinations_id')
    df = transform_seller_address_state(df)
    df = transform_seller_address_city_id(df)

    return df