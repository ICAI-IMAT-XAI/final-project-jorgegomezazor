import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
from typing import Tuple

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Entrena un clasificador XGBoost con hiperparámetros optimizados para HELOC.
    
    Args:
        X_train: Conjunto de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        
    Returns:
        xgb.XGBClassifier: Modelo entrenado.
    """
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.15,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate(
    model: xgb.XGBClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[float, float]:
    """
    Realiza una evaluación exhaustiva del modelo utilizando AUC, 
    Accuracy y métricas por clase.
    
    Args:
        model: Modelo entrenado.
        X_test: Conjunto de prueba.
        y_test: Etiquetas reales de prueba.
        
    Returns:
        Tuple: (AUC score, Accuracy score).
    """
    # Predicción de probabilidades y clases
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    # Cálculo de métricas
    auc = roc_auc_score(y_test, probs)
    accuracy = (preds == y_test).mean()
    
    # Reporte de resultados
    print("-" * 30)
    print("RESUMEN DE EVALUACIÓN")
    print("-" * 30)
    print(f"AUC Score: {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print("\nDetalle por Clase (Classification Report):")
    print(classification_report(y_test, preds, target_names=['Good (0)', 'Bad (1)']))
    print("-" * 30)
    
    return auc, accuracy