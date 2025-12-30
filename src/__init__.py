"""
Módulo de inicialización del paquete src.
Centraliza las funcionalidades de preprocesamiento, entrenamiento, 
explicabilidad y validación (Sanity Checks) para el dataset HELOC.
"""

from .preprocess import load_and_preprocess
from .train import train_model, evaluate
from .explain import (
    get_global_explanations, 
    get_local_explanation, 
    generate_counterfactuals
)
from .sanity_checks import (
    run_model_parameter_randomization, 
    check_feature_perturbation
)

__all__ = [
    'load_and_preprocess',
    'train_model',
    'evaluate',
    'get_global_explanations',
    'get_local_explanation',
    'generate_counterfactuals',
    'run_model_parameter_randomization',
    'check_feature_perturbation'
]