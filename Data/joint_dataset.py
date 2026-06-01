import pandas as pd
import numpy as np
import json

def parsear_linea_metadata(line: str) -> dict:
    """
    Somete cada línea de metadatos a un análisis de partición adaptativo.
    """
    tokens = [t.strip() for t in line.split(',')]
    if len(tokens) < 2:
        return None
    
    pkey = tokens[0]
    temp_raw = tokens[-1]
    
    # Vocabulario de control para identificar la columna central 'folder'
    folders_conocidos = {
        'Phosphate', 'Sulfate', 'Oxide', 'Metal', 'Chlorite', 'Kaolinite-Serpentine', 
        'Mixtures', 'Glass', 'Beryl', 'Terrestrial Samples', 'Alunite', 'Olivine', 
        'Pyroxene', 'Amphibole', 'Feldspar', 'Mica', 'Clay', 'Carbonate', 'Sulfide', 'N/A'
    }
    
    # Intentar anclaje por coincidencia de carpeta
    folder_idx = -1
    for i, token in enumerate(tokens):
        if token in folders_conocidos and i > 0:
            folder_idx = i
            break
            
    if folder_idx != -1:
        # Reconstrucción asociativa de Dana Class (absorbe comas internas)
        dana_class = ",".join(tokens[1:folder_idx])
        folder = tokens[folder_idx]
        resto = tokens[folder_idx+1:-1]
        
        # Segmentación de Sample Name y Owner
        if len(resto) >= 3:
            sample_name = resto[0]
            owner = ",".join(resto[1:])
        elif len(resto) == 2:
            sample_name = resto[0]
            owner = resto[1]
        else:
            sample_name = resto[0] if resto else "N/A"
            owner = "N/A"
    else:
        # Fallback: Escaneo inverso si la carpeta no está en el vocabulario de control
        # Asume estructura estándar desde la derecha: [Sample, Owner_Last, Owner_First, Temp] -> len=4 en el resto
        resto_inverso = tokens[1:-1]
        if len(resto_inverso) >= 4:
            dana_class = resto_inverso[0]
            folder = resto_inverso[1]
            sample_name = resto_inverso[2]
            owner = ",".join(resto_inverso[3:])
        else:
            dana_class = ",".join(resto_inverso)
            folder, sample_name, owner = "N/A", "N/A", "N/A"
            
    return {
        'pkey': pkey,
        'Dana Class': dana_class,
        'folder': folder,
        'Sample Name': sample_name,
        'Owner/Source': owner,
        'Temperature (K)': temp_raw
    }

def generar_dataset_combinado_corregido(ruta_metadata: str, ruta_espectros: str, ruta_salida: str) -> pd.DataFrame:
    """
    Integra la carga limpia de columnas estructuradas de metadatos con matrices espectrales.
    """
    # 1. Extracción y estructuración de Metadatos
    registros_meta = []
    with open(ruta_metadata, 'r', encoding='utf-8', errors='replace') as f:
        f.readline() # Omitir header
        for line in f:
            line = line.strip()
            if not line: 
                continue
            res = parsear_linea_metadata(line)
            if res:
                registros_meta.append(res)
                
    meta_df = pd.DataFrame(registros_meta)
    
    # Forzar consistencia numérica en la temperatura termodinámica
    meta_df['Temperature (K)'] = pd.to_numeric(meta_df['Temperature (K)'], errors='coerce')

    # 2. Lectura matricial de espectros y propagación (Forward Fill)
    col_names = ['pkey', 'axis'] + [f'pt_{i}' for i in range(512)]
    spectra_df = pd.read_csv(ruta_espectros, names=col_names, skiprows=1, low_memory=False)
    spectra_df['pkey'] = spectra_df['pkey'].astype(str).str.strip().replace('nan', np.nan).ffill()

    # 3. Pivotado de canales de hardware a arreglos densos
    pts_cols = [col for col in spectra_df.columns if col.startswith('pt_')]
    spectra_dict = {}

    for (pkey, axis), group in spectra_df.groupby(['pkey', 'axis']):
        arr = [float(x) for x in group[pts_cols].iloc[0] if pd.notnull(x)]
        if pkey not in spectra_dict: 
            spectra_dict[pkey] = {}
        spectra_dict[pkey][axis] = arr

    spectra_lists = pd.DataFrame.from_dict(spectra_dict, orient='index')
    spectra_lists.index.name = 'pkey'
    spectra_lists = spectra_lists.reset_index()

    # 4. Cruce Relacional Determinista
    meta_df['pkey'] = meta_df['pkey'].astype(str).str.strip()
    spectra_lists['pkey'] = spectra_lists['pkey'].astype(str).str.strip()
    
    merged_df = pd.merge(meta_df, spectra_lists, on='pkey', how='inner')

    # 5. Serialización compatible con almacenamiento plano
    if 'Velocity (mm/s)' in merged_df.columns:
        merged_df['Velocity (mm/s)'] = merged_df['Velocity (mm/s)'].apply(json.dumps)
    if 'Intensity' in merged_df.columns:
        merged_df['Intensity'] = merged_df['Intensity'].apply(json.dumps)

    merged_df.to_csv(ruta_salida, index=False)
    return merged_df

import sys
from pathlib import Path

# (Inserte aquí las funciones parsear_linea_metadata y generar_dataset_combinado_corregido)

if __name__ == '__main__':
    # 1. Anclaje absoluto basado en el descriptor del script
    BASE_DIR = Path(__file__).resolve().parent
    
    # 2. Definición del subdirectorio (Sensible a mayúsculas)
    dir_datos = BASE_DIR 
    
    ruta_meta = dir_datos / "metadata.csv"
    ruta_spec = dir_datos / "spectra.csv"
    ruta_out = dir_datos / "data.csv"
    
    # 3. Invocación controlada
    try:
        df_unificado = generar_dataset_combinado_corregido(
            ruta_metadata=str(ruta_meta), 
            ruta_espectros=str(ruta_spec), 
            ruta_salida=str(ruta_out)
        )
        print(f"Operación finalizada. Dimensionalidad del tensor resultante: {df_unificado.shape}")
    except FileNotFoundError as e:
        print(f"Excepción topológica: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error en tiempo de ejecución: {e}", file=sys.stderr)
        sys.exit(1)