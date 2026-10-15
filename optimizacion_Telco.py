import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

from preprocesamiento import preprocesar_telco
from balanceo_y_split import preparar_datos_balanceados

def optimizar_telco():
    # 1. Preprocesar y balancear
    X_telco, y_telco = preprocesar_telco("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = preparar_datos_balanceados(X_telco, y_telco)
    
    y_train_num = y_train.map({'Yes': 1, 'No': 0, 1: 1, 0: 0}) if hasattr(y_train, 'map') else y_train
    y_test_num = y_test.map({'Yes': 1, 'No': 0, 1: 1, 0: 0}) if hasattr(y_test, 'map') else y_test

    # 2. Rejilla de Hiperparámetros para Regresión Logística
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        param_grid=param_grid,
        scoring='recall', # Enfocado en maximizar detección de Churn
        cv=5,
        n_jobs=-1
    )
    
    print("=== OPTIMIZANDO HIPERPARÁMETROS (TELCO) ===")
    grid.fit(X_train, y_train_num)
    
    print(f"Mejores Parámetros: {grid.best_params_}")
    print(f"Mejor Recall en Validación Cruzada: {grid.best_score_:.4f}")
    
    # Evaluar mejor modelo en Test
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\nReporte en Conjunto de Prueba (Test):")
    print(classification_report(y_test_num, y_pred, target_names=['No Churn', 'Churn']))
    print(f"ROC-AUC Final: {roc_auc_score(y_test_num, y_proba):.4f}")

if __name__ == "__main__":
    optimizar_telco()