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

from preprocesamiento import preprocesar_telco
from balanceo_y_split import preparar_datos_balanceados

def evaluar_modelos_telco():
    ruta_telco = "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # 1. Preprocesar
    X_telco, y_telco = preprocesar_telco(ruta_telco)
    
    # 2. Balancear
    X_train, X_test, y_train, y_test = preparar_datos_balanceados(X_telco, y_telco)
    
    # Estandarizar etiquetas a 'Yes' / 'No'
    y_train = y_train.map({1: 'Yes', 0: 'No', 'Yes': 'Yes', 'No': 'No'}) if hasattr(y_train, 'map') else y_train
    y_test = y_test.map({1: 'Yes', 0: 'No', 'Yes': 'Yes', 'No': 'No'}) if hasattr(y_test, 'map') else y_test

    modelos = {
        'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    resultados = []
    
    y_train_num = y_train.map({'Yes': 1, 'No': 0})
    y_test_num = y_test.map({'Yes': 1, 'No': 0})
    
    for nombre, modelo in modelos.items():
        if nombre == 'XGBoost':
            modelo.fit(X_train, y_train_num)
            y_pred_num = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]
            y_pred = pd.Series(y_pred_num).map({1: 'Yes', 0: 'No'})
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]
        
        resultados.append({
            'Modelo': nombre,
            'Precision': round(precision_score(y_test, y_pred, pos_label='Yes'), 4),
            'Recall': round(recall_score(y_test, y_pred, pos_label='Yes'), 4),
            'F1-Score': round(f1_score(y_test, y_pred, pos_label='Yes'), 4),
            'ROC-AUC': round(roc_auc_score(y_test_num, y_proba), 4)
        })
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f'Matriz de Confusión Telco - {nombre}')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.show() # Se pausará hasta que cierres la ventana manualmente
        
    df_res = pd.DataFrame(resultados)
    print("\n=== RESUMEN COMPARATIVO (TELCO CHURN) ===")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    evaluar_modelos_telco()