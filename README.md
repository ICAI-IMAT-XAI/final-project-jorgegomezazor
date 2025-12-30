# Proyecto Final: Interpretabilidad en HELOC

Este repositorio contiene el proyecto final de la asignatura de Ética y Explicabilidad de la IA. Se analiza el riesgo crediticio mediante un modelo XGBoost y técnicas avanzadas de explicabilidad vistas en clase.

## Cómo ejecutar el proyecto

Sigue estos pasos para configurar el entorno y ver los resultados:

1. **Crear el entorno virtual:**

   ```bash
   python -m venv .venv
    ```

2. **Activar el entorno:**


* En Windows: `.venv\Scripts\activate`
* En Mac/Linux: `source .venv/bin/activate`


3. **Instalar dependencias:**

    ```bash
    pip install -r requirements.txt
    ```


4. **Ejecución:**

Abre el archivo `notebooks/pf.ipynb` y selecciona "Run All" (Ejecutar todo). Asegúrate de que el kernel seleccionado sea el de tu `.venv`.

---

## Contenido del proyecto:
**`notebooks/pf.ipynb`**: Notebook principal con todo el flujo de ejecución.
**`src/`**: Código modularizado (preprocesamiento, entrenamiento, explicabilidad y sanity checks)
