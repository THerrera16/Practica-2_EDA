import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Configuración de estilo idéntica
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})

# Carpeta para guardar las imágenes de esta actividad
carpeta_salida = "Desbalance Titanic"
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar dataset del Titanic (ajusta el nombre del archivo si es necesario)
ruta_titanic = "titanic-Dataset.csv"
if not os.path.exists(ruta_titanic):
    # Intento alternativo de nombre común
    ruta_titanic = "titanic.csv"

df_titanic = pd.read_csv(ruta_titanic)

# Preprocesamiento básico para aislar características numéricas y la variable objetivo
target_col = 'Survived' if 'Survived' in df_titanic.columns else 'target'
cols_num = df_titanic.select_dtypes(include=[np.number]).columns.tolist()
if target_col in cols_num:
    cols_num.remove(target_col)
if 'PassengerId' in cols_num:
    cols_num.remove('PassengerId')

# Rellenar nulos básicos para evitar errores de modelado
df_titanic[cols_num] = df_titanic[cols_num].fillna(df_titanic[cols_num].median())
df_titanic = df_titanic.dropna(subset=[target_col])

X = df_titanic[cols_num]
y = df_titanic[target_col]

print("==========================================")
print("ANÁLISIS DE DESBALANCE: TITANIC")
print("==========================================")

# 1. CÁLCULO DE MÉTRICAS DE DESBALANCE
conteo = y.value_counts()
porcentajes = y.value_counts(normalize=True) * 100

print("\n--- Conteo de observaciones por clase ---")
print(conteo)
print("\n--- Porcentaje por clase ---")
print(porcentajes)

mayoritaria = conteo.idxmax()
minoritaria = conteo.idxmin()
IR = conteo[mayoritaria] / conteo[minoritaria]

print(f"\nRazón de desbalance (IR): {IR:.2f}")
if IR > 1.5:
    print("-> Resultado: El dataset está DESBALANCEADO (IR > 1.5)[cite: 3].\n")
else:
    print("-> Resultado: El dataset está balanceado.\n")

# Gráfica 1: Distribución de Clases con anotaciones de porcentajes
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df_titanic, x=target_col, palette='Blues_d')
plt.title("1. Distribución de Clases (Titanic)", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Supervivencia (0 = No, 1 = Sí)")
plt.ylabel("Frecuencia")

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())} ({p.get_height()/len(df_titanic)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "1_distribucion_clases_titanic.png"), dpi=300)
plt.close()


# División de datos (Train/Test estratificado)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 2. IMPLEMENTACIÓN Y EVALUACIÓN DE LAS 5 ESTRATEGIAS
estrategias = {
    "1. Sin balanceo (Línea base)": (X_train, y_train, None),
    "2. Ponderación de clases": (X_train, y_train, "balanced"),
    "3. Sobremuestreo aleatorio": *RandomOverSampler(random_state=42).fit_resample(X_train, y_train), "none",
    "4. Submuestreo aleatorio": *RandomUnderSampler(random_state=42).fit_resample(X_train, y_train), "none",
    "5. SMOTE": *SMOTE(random_state=42).fit_resample(X_train, y_train), "none"
}

print("==========================================")
print("EVALUACIÓN DE MODELOS POR ESTRATEGIA")
print("==========================================")

for nombre, datos in estrategias.items():
    if len(datos) == 3:
        X_tr, y_tr, cw = datos
        modelo = LogisticRegression(class_weight=cw, random_state=42, max_iter=1000)
    else:
        X_tr, y_tr, _ = datos
        modelo = LogisticRegression(random_state=42, max_iter=1000)
    
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_test)
    
    print(f"\n--- ESTRATEGIA: {nombre} ---")
    print(classification_report(y_test, y_pred, zero_division=0))

print(f"\n¡Proceso de Titanic finalizado! Imágenes guardadas en la carpeta: '{carpeta_salida}'.")