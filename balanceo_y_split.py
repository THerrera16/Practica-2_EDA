import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# Importamos la función desde el archivo del Paso 1
from preprocesamiento import preprocesar_telco, preprocesar_titanic

def preparar_datos_balanceados(X, y, test_size=0.20, random_state=42):
    """
    Realiza la partición estratificada de datos y aplica SMOTE 
    únicamente al conjunto de entrenamiento.
    """
    # 1. División Train / Test estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print("--- Distribución Original (Train) ---")
    print(Counter(y_train))
    
    # 2. Aplicación de SMOTE únicamente en Train
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("\n--- Distribución Tras SMOTE (Train) ---")
    print(Counter(y_train_res))
    print(f"\nConjunto de Prueba (Test) intacto: {len(y_test)} muestras.")
    
    return X_train_res, X_test, y_train_res, y_test

# BLOQUE DE EJECUCIÓN CON DATOS REALES
if __name__ == "__main__":
    print("=== PROCESANDO TELCO CUSTOMER CHURN ===")
    ruta_telco = "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    X_telco, y_telco = preprocesar_telco(ruta_telco)
    X_train_res, X_test, y_train_res, y_test = preparar_datos_balanceados(X_telco, y_telco)