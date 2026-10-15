import os
import matplotlib
# IMPORTANTE: Forzar el backend no interactivo para asegurar que guarde las imágenes sin interfaz gráfica
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Configuración de estilo
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})

carpeta_salida = "Desbalance Titanic"
os.makedirs(carpeta_salida, exist_ok=True)
print(f"-> Carpeta creada/verificada: '{carpeta_salida}'")

# Ruta correcta basada en la carpeta 'datasets'
ruta_titanic = os.path.join("datasets", "Titanic-Dataset.csv")
if not os.path.exists(ruta_titanic):
    print(f"Error crítico: No se encuentra el archivo en '{ruta_titanic}'")
    exit()

df_titanic = pd.read_csv(ruta_titanic)

target_col = 'Survived' if 'Survived' in df_titanic.columns else 'target'
cols_num = df_titanic.select_dtypes(include=[np.number]).columns.tolist()
if target_col in cols_num:
    cols_num.remove(target_col)
if 'PassengerId' in cols_num:
    cols_num.remove('PassengerId')

df_titanic[cols_num] = df_titanic[cols_num].fillna(df_titanic[cols_num].median())
df_titanic = df_titanic.dropna(subset=[target_col])

X = df_titanic[cols_num]
y = df_titanic[target_col]

print("==========================================")
print("ANÁLISIS DE DESBALANCE: TITANIC")
print("==========================================")

# Cálculo de métricas de desbalance
conteo = y.value_counts()
porcentajes = y.value_counts(normalize=True) * 100

print("\n--- Conteo de observaciones por clase ---")
print(conteo)

mayoritaria = conteo.idxmax()
minoritaria = conteo.idxmin()
IR = conteo[mayoritaria] / conteo[minoritaria]
print(f"Razón de desbalance (IR): {IR:.2f}")

# 1. Gráfica de distribución de clases
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
ruta_img1 = os.path.join(carpeta_salida, "1_distribucion_clases_titanic.png")
plt.savefig(ruta_img1, dpi=300)
plt.close()
print(f"-> Guardada imagen: {ruta_img1}")

# División de datos y remuestreo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_over, y_train_over = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
X_train_under, y_train_under = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)

estrategias = {
    "2. Sin balanceo": (X_train, y_train, None),
    "3. Ponderación de clases": (X_train, y_train, "balanced"),
    "4. Sobremuestreo": (X_train_over, y_train_over, None),
    "5. Submuestreo": (X_train_under, y_train_under, None),
    "6. SMOTE": (X_train_smote, y_train_smote, None)
}

print("\n==========================================")
print("EVALUACIÓN Y GENERACIÓN DE MATRICES")
print("==========================================")

for idx, (nombre, (X_tr, y_tr, cw)) in enumerate(estrategias.items(), start=2):
    modelo = LogisticRegression(class_weight=cw, random_state=42, max_iter=1000)
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_test)
    
    print(f"\n--- Estrategia procesada: {nombre} ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Generar y guardar matriz de confusión como imagen
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Resultado - {nombre}", fontsize=12, pad=10, fontweight='bold')
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    
    nombre_archivo = f"{idx}_matriz_resultado_{nombre.lower().replace(' ', '_').replace('.', '')}.png"
    ruta_img = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_img, dpi=300)
    plt.close()
    print(f"-> Guardada imagen: {ruta_img}")

print(f"\n¡Proceso de Titanic finalizado con éxito! Revisa la carpeta '{carpeta_salida}'.")