import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from typing import Union

def run_model_parameter_randomization(
    model: xgb.XGBClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> float:
    """
    Realiza el test de randomización de parámetros del modelo. 
    Verifica si las explicaciones (SHAP) son sensibles a los pesos aprendidos 
    comparándolas con un modelo entrenado con etiquetas aleatorias.
    
    Args:
        model: Modelo XGBoost entrenado con datos reales.
        X_test: Conjunto de variables predictoras de prueba.
        y_test: Etiquetas reales de prueba.
        
    Returns:
        float: Coeficiente de correlación de Pearson entre ambas importancias.
    """
    # 1. Obtención de importancia SHAP del modelo real
    explainer = shap.TreeExplainer(model)
    orig_shap_values = explainer.shap_values(X_test)
    orig_importance = np.abs(orig_shap_values).mean(axis=0)
    
    # 2. Entrenamiento del modelo de control (Ruido)
    # Se utilizan etiquetas permutadas aleatoriamente para romper el aprendizaje
    print("Iniciando Sanity Check: Entrenando modelo de control con etiquetas aleatorias...")
    noise_model = xgb.XGBClassifier(
        n_estimators=5, 
        max_depth=2, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    noise_model.fit(X_test, np.random.permutation(y_test))
    
    # 3. Obtención de importancia SHAP del modelo de ruido
    noise_explainer = shap.TreeExplainer(noise_model)
    noise_shap_values = noise_explainer.shap_values(X_test)
    noise_importance = np.abs(noise_shap_values).mean(axis=0)
    
    # 4. Cálculo de correlación de estabilidad
    correlation = np.corrcoef(orig_importance, noise_importance)[0, 1]
    
    print(f"\n--- Sanity Check: Randomización de Parámetros ---")
    print(f"Correlación entre modelo real y modelo ruido: {correlation:.4f}")
    
    # Un valor bajo (< 0.5) indica que el método XAI es fiel al aprendizaje del modelo
    passed = correlation < 0.5
    print(f"Resultado: {'PASADO' if passed else 'FALLIDO'}\n")
    
    return correlation

def check_feature_perturbation(
    model: xgb.XGBClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    feature_name: str
) -> float:
    """
    Evalúa la sensibilidad del modelo ante la perturbación de una característica específica.
    Si la variable es relevante, el rendimiento del modelo (AUC) debe disminuir al permutarla.
    
    Args:
        model: Modelo entrenado.
        X_test: Conjunto de prueba.
        y_test: Etiquetas reales.
        feature_name: Nombre de la columna a permutar.
        
    Returns:
        float: Magnitud de la caída en el AUC.
    """
    # 1. Cálculo del rendimiento base (Baseline)
    baseline_probs = model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_probs)
    
    # 2. Perturbación de la variable: Permutación aleatoria de sus valores
    X_permuted = X_test.copy()
    X_permuted[feature_name] = np.random.permutation(X_permuted[feature_name].values)
    
    # 3. Evaluación del modelo con la variable degradada
    permuted_probs = model.predict_proba(X_permuted)[:, 1]
    permuted_auc = roc_auc_score(y_test, permuted_probs)
    
    performance_drop = baseline_auc - permuted_auc
    
    print(f"--- Sanity Check: Importancia por Permutación ({feature_name}) ---")
    print(f"Baseline AUC:      {baseline_auc:.4f}")
    print(f"AUC tras permutar: {permuted_auc:.4f}")
    print(f"Caída de performance: {performance_drop:.4f}\n")
    
    return performance_drop