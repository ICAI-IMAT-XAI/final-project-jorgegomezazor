import pandas as pd
import numpy as np
import shap
import dice_ml
from typing import Tuple, List, Any

def get_global_explanations(
    model: Any, 
    X: pd.DataFrame
) -> Tuple[shap.TreeExplainer, np.ndarray]:
    """
    Genera el explicador y los valores SHAP globales para un modelo basado en árboles.
    
    Args:
        model: Modelo entrenado (XGBoost, RandomForest, etc.).
        X: Conjunto de datos para calcular las explicaciones.
        
    Returns:
        Tuple: (Explainer objeto, SHAP values).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    return explainer, shap_values

def get_local_explanation(explainer: shap.TreeExplainer, instance: pd.DataFrame) -> np.ndarray:
    """
    Calcula los valores SHAP para una instancia específica.
    
    Args:
        explainer: Objeto TreeExplainer ya inicializado.
        instance: Fila del DataFrame (una sola observación).
        
    Returns:
        np.ndarray: Valores SHAP de la instancia.
    """
    return explainer(instance)

def generate_counterfactuals(
    model: Any, 
    df_train: pd.DataFrame, 
    instance: pd.DataFrame, 
    target_column: str,
    desired_class: int = 0, 
    num_cfs: int = 5,
    continuous_features: List[str] = None,
    features_to_vary: List[str] = None
) -> dice_ml.counterfactual_explanations.CounterfactualExplanations:
    """
    Genera explicaciones contrafactuales (What-if) utilizando DiCE.
    
    Args:
        model: Modelo entrenado con interfaz sklearn.
        df_train: DataFrame original de entrenamiento (incluyendo el target).
        instance: Instancia para la que se busca el contrafactual.
        target_column: Nombre de la variable objetivo.
        desired_class: Clase deseada (ej: 0 para 'Good' credit).
        num_cfs: Número de ejemplos contrafactuales a generar.
        continuous_features: Lista de nombres de columnas continuas.
        features_to_vary: Lista de características que se permiten variar en el contrafactual.
        
    Returns:
        CounterfactualExplanations: Objeto con los contrafactuales generados.
    """
    # 1. Configuración de la interfaz de datos de DiCE
    d = dice_ml.Data(
        dataframe=df_train,
        continuous_features=continuous_features or list(df_train.drop(target_column, axis=1).columns),
        outcome_name=target_column
    )
    
    # 2. Configuración del modelo para DiCE (Backend sklearn/XGBoost)
    m = dice_ml.Model(model=model, backend="sklearn")
    
    # 3. Inicialización del método y generación
    exp = dice_ml.Dice(d, m, method="random") # 'random' es robusto para datos tabulares
    
    dice_exp = exp.generate_counterfactuals(
        instance, 
        total_CFs=num_cfs, 
        desired_class=desired_class,
        features_to_vary=features_to_vary or 'all'
    )
    
    return dice_exp
