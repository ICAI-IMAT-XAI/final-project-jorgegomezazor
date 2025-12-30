import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_and_preprocess(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Carga el dataset HELOC y realiza el preprocesamiento básico: 
    codificación del target y división en conjuntos de entrenamiento y prueba.
    
    Args:
        path (str): Ruta al archivo CSV del dataset.
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    # Carga de datos
    df = pd.read_csv(path)
    
    # Codificación de la variable objetivo (Bad: 1, Good: 0)
    df['target'] = df['RiskPerformance'].map({'Bad': 1, 'Good': 0})
    
    # Separación de variables predictoras y objetivo
    X = df.drop(['RiskPerformance', 'target'], axis=1)
    y = df['target']
    
    # Eliminación de filas con valores faltantes o con todos los valores a -9
    X = X[(X != -9).all(axis=1)]
    y = y[X.index]
    X = X.dropna()
    y = y[X.index]
    
    # División estratificada para mantener la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test