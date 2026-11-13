import os
import re
import warnings

import pandas as pd

# pyBKT 1.4.x inspects private sklearn metric namespaces during import.
# Newer scikit-learn versions expose those namespaces differently, so we wrap the
# real metric functions in a NumPy-coercion layer and convert invalid dummy checks
# into TypeError so pyBKT can ignore unsupported sklearn metrics in a clean way.
def _patch_pyBKT_sklearn_compat():
    try:
        import numpy as np
        import sklearn.metrics as sk_metrics
    except ImportError:
        return

    class CompatMetricNamespace:
        def __getattr__(self, name):
            if not re.search(r'(_loss|_score|_error)$', name):
                raise AttributeError(name)
            real_metric = getattr(sk_metrics, name)

            def wrapped(*args, **kwargs):
                coerced_args = []
                for arg in args:
                    if isinstance(arg, (list, tuple, np.ndarray)):
                        coerced_args.append(np.asarray(arg))
                    else:
                        coerced_args.append(arg)
                try:
                    return real_metric(*coerced_args, **kwargs)
                except (TypeError, ValueError):
                    raise TypeError

            return wrapped

        def __dir__(self):
            return [
                name for name in dir(sk_metrics)
                if re.search(r'(_loss|_score|_error)$', name)
            ]

    sk_metrics._classification = CompatMetricNamespace()
    sk_metrics._regression = CompatMetricNamespace()


_patch_pyBKT_sklearn_compat()

from pyBKT.models import Model

DATA_PATH = "synthetic_bkt_dataset.csv"
DEFAULTS = {
    'user_id': 'student_id',
    'skill_name': 'skill_name',
    'correct': 'correct',
}


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No se encontró el dataset '{DATA_PATH}'. Primero ejecuta generar_datos.py."
        )

    # 1. Cargar el dataset transaccional que generamos en la Fase 1
    df = pd.read_csv(DATA_PATH)

    # 2. Inicializar el modelo BKT
    # Usar una semilla (seed) garantiza que los resultados sean reproducibles en tu portafolio
    model = Model(seed=42, num_fits=1, defaults=DEFAULTS, parallel=False)
    cv_model = Model(seed=42, num_fits=1, defaults=DEFAULTS, parallel=False)

    # 3. Entrenamiento del Modelo (Fit)
    print("Entrenando el modelo BKT...")
    # Mapeamos las columnas de nuestro CSV a las que requiere la librería
    model.fit(data=df, defaults=DEFAULTS, parallel=False)

    # 4. Evaluación Robusta (Cross-Validation)
    # Aplicamos validación cruzada para asegurar que el modelo no esté sobreajustado
    print("\nEjecutando validación cruzada (3 folds)...")
    warnings.filterwarnings("ignore", category=Warning, module="sklearn.metrics._ranking")
    metrics_df = cv_model.crossvalidate(
        data=df,
        folds=3,
        defaults=DEFAULTS,
        parallel=False,
        metric=['rmse', 'auc'],
    )

    # Calculamos el promedio del Área Bajo la Curva (AUC) y el Error Cuadrático Medio (RMSE)
    rmse_mean = metrics_df['rmse'].mean()
    auc_mean = metrics_df['auc'].mean()
    print(f"Métricas Generales -> RMSE: {rmse_mean:.4f} | AUC: {auc_mean:.4f}")

    # 5. Interpretabilidad: Visualizar lo que aprendió el modelo
    # Para un científico de datos en EdTech, estos parámetros son oro puro
    print("\nParámetros calculados por micro-habilidad (Prior, Learn, Guess, Slip):")
    learned_params = model.params()
    print(learned_params)

    # 6. Predicción Predictiva (Inferencia)
    # Calcula la probabilidad exacta de que el estudiante acierte en su próximo intento
    predictions = model.predict(data=df)

    print("\nMuestra de probabilidades predichas (correct_predictions = % de probabilidad de acierto):")
    # Mostramos los últimos 10 intentos registrados
    print(predictions[['student_id', 'skill_name', 'correct', 'correct_predictions']].tail(10))


if __name__ == "__main__":
    main()