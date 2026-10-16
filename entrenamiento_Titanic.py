import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score
)

from preprocesamiento import preprocesar_titanic
from balanceo_y_split import preparar_datos_balanceados

def evaluar_modelos_titanic():
    ruta_titanic = "datasets/Titanic-Dataset.csv"
    
    # 1. Preprocesar
    X_titanic, y_titanic = preprocesar_titanic(ruta_titanic)
    
    # 2. Balancear
    X_train, X_test, y_train, y_test = preparar_datos_balanceados(X_titanic, y_titanic)

    modelos = {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    resultados = []
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        
        resultados.append({
            'Modelo': nombre,
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4)
        })
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
        plt.title(f'Matriz de Confusión Titanic - {nombre}')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.show() # Se pausará hasta que cierres la ventana manualmente
        
    df_res = pd.DataFrame(resultados)
    print("\n=== RESUMEN COMPARATIVO (TITANIC SURVIVED) ===")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    evaluar_modelos_titanic()