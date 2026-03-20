import pandas as pd

def import_dataset(asset='gc1', format='dia'):
    """
    Importa un dataset de texto y normaliza las columnas de fecha y hora.

    Parámetros:
    - asset (str): prefijo del archivo de activo (ej. 'gc1').
    - format (str): sufijo de formato de archivo (ej. 'dia' o 'min').

    El archivo esperado está en `../data/{asset}{format}.txt` y debe contener
    al menos las columnas 'DTYYYYMMDD' y 'TIME'.

    El proceso realiza:
    1. Lectura del CSV.
    2. Conversión de 'DTYYYYMMDD' a tipo datetime.
    3. Normalización de 'TIME' (relleno a 6 dígitos) y conversión a time.
    4. Creación de la columna 'DATETIME' uniendo fecha y hora.

    Retorna:
    - pd.DataFrame con las columnas originales y la nueva columna 'DATETIME'.
    """
    file_path = f'../Data/{asset}{format}.txt'
    df = pd.read_csv(file_path)
    # Normalizar nombres de columnas a minúsculas
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'vol': 'volume'
    })
    # Convertir columna de fecha (formato YYYYMMDD) a datetime
    df['dtyyyymmdd'] = pd.to_datetime(df['dtyyyymmdd'], format='%Y%m%d')
    # Asegurar que time tenga 6 dígitos (HHMMSS) y convertir a tipo time
    df['time'] = df['time'].astype(str).str.zfill(6)
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S').dt.time
    # Unir fecha y hora en una sola columna de tipo datetime
    df['datetime'] = pd.to_datetime(df['dtyyyymmdd'].astype(str) + ' ' + df['time'].astype(str))
    df = df.sort_values('datetime')
    return df

def load_future(asset='gc1', format='all'):
    """_summary_

    Args:
        asset (str, optional): _description_. Defaults to 'gc1'.
        format (str, optional): _description_. Defaults to 'all'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if format == 'all':
        df_dia = import_dataset(asset, 'dia')
        df_min = import_dataset(asset, 'min')
    elif format == 'dia':
        df_dia = import_dataset(asset, 'dia')
    elif format == 'min':
        df_min = import_dataset(asset, 'min')
    else:
        raise ValueError("Formato no reconocido. Use 'dia', 'min' o 'all'.")
    specs = pd.read_json("../Data/futuros_specs.json")
    spec = specs[asset[:2].upper()]
    return spec, df_dia if format in ['dia', 'all'] else None, df_min if format in ['min', 'all'] else None