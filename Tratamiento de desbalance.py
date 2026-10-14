import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 1. Cargar tu dataset (ajusta la ruta según tu carpeta 'data/')
# df = pd.read_csv('../data/tu_dataset.csv')

# Simularemos un dataframe para que el script sea ejecutable de ejemplo:
np.random.seed(42)
X_dummy = np.random.randn(1000, 5)
# Creamos un desbalance extremo (ej. 95% clase 0, 5% clase 1)
y_dummy = np.array([0]*950 + [1]*5)
df = pd.DataFrame(X_dummy, columns=['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'])
df['target'] = y_dummy

# Separar características (X) y variable objetivo (y)
X = df.drop(columns=['target'])
y = df['target']

# 2. Cálculo de métricas de desbalance (Paso 1)
conteo_clases = y.value_counts()
porcentaje_clases = y.value_counts(normalize=True) * 100

print("--- Conteo de observaciones por clase ---")
print(conteo_clases)
print("\n--- Porcentaje por clase ---")
print(porcentaje_clases)

# Identificar clase mayoritaria y minoritaria
clase_mayoritaria = conteo_clases.idxmax()
clase_minoritaria = conteo_clases.idxmin()

N_mayoritaria = conteo_clases[clase_mayoritaria]
N_minoritaria = conteo_clases[clase_minoritaria]

# Calcular el Imbalance Ratio (IR)
IR = N_mayoritaria / N_minoritaria
print(f"\nRazón de desbalance (IR): {IR:.2f}")

if IR > 1.5:
    print("Resultado: El dataset se considera DESBALANCEADO (IR > 1.5)[cite: 3].\n")
else:
    print("Resultado: El dataset está balanceado.\n")

# División de datos (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Implementación de Estrategias de Tratamiento (Paso 2)

estrategias = {
    "1. Sin balanceo (Línea base)": (X_train, y_train, None),
    "2. Ponderación de clases": (X_train, y_train, "balanced"),
    "3. Sobremuestreo aleatorio": *RandomOverSampler(random_state=42).fit_resample(X_train, y_train), "none",
    "4. Submuestreo aleatorio": *RandomUnderSampler(random_state=42).fit_resample(X_train, y_train), "none",
    "5. SMOTE": *SMOTE(random_state=42).fit_resample(X_train, y_train), "none"
}

# Evaluar cada estrategia con un modelo de Regresión Logística
for nombre, datos in estrategias.items():
    if len(datos) == 3:
        X_tr, y_tr, class_weight = datos
        modelo = LogisticRegression(class_weight=class_weight, random_state=42, max_iter=1000)
    else:
        X_tr, y_tr, _ = datos
        modelo = LogisticRegression(random_state=42, max_iter=1000)
    
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_test)
    
    print(f"================ ESTRATEGIA: {nombre} ================")
    print(classification_report(y_test, y_pred, zero_division=0))
