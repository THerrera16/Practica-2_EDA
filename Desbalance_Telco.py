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

carpeta_salida = "Desbalance Telco"
os.makedirs(carpeta_salida, exist_ok=True)
print(f"-> Carpeta creada/verificada: '{carpeta_salida}'")

# Ruta correcta basada en la carpeta 'datasets'
ruta_telco = os.path.join("datasets", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not os.path.exists(ruta_telco):
    print(f"Error crítico: No se encuentra el archivo en '{ruta_telco}'")
    exit()

df_telco = pd.read_csv(ruta_telco)

df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'].astype(str).str.strip(), errors='coerce')
df_telco['TotalCharges'].fillna(df_telco['TotalCharges'].median(), inplace=True)
df_telco['tenure'].fillna(df_telco['tenure'].median(), inplace=True)
df_telco['MonthlyCharges'].fillna(df_telco['MonthlyCharges'].median(), inplace=True)

df_telco['Target'] = df_telco['Churn'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
df_telco['Abandono_ES'] = df_telco['Target'].map({0: 'Permanece (No)', 1: 'Abandona (Sí)'})

cols_num_telco = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df_telco[cols_num_telco]
y = df_telco['Target']

print("==========================================")
print("ANÁLISIS DE DESBALANCE: TELCO CHURN")
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
ax = sns.countplot(data=df_telco, x='Abandono_ES', palette=['#2b5c8f', '#d95f02'])
plt.title("1. Distribución de Clases (Telco Churn)", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Estado del Cliente")
plt.ylabel("Frecuencia")

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())} ({p.get_height()/len(df_telco)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
ruta_img1 = os.path.join(carpeta_salida, "1_distribucion_clases_telco.png")
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.title(f"Resultado - {nombre}", fontsize=12, pad=10, fontweight='bold')
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    
    nombre_archivo = f"{idx}_matriz_resultado_{nombre.lower().replace(' ', '_').replace('.', '')}.png"
    ruta_img = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_img, dpi=300)
    plt.close()
    print(f"-> Guardada imagen: {ruta_img}")

print(f"\n¡Proceso de Telco finalizado con éxito! Revisa la carpeta '{carpeta_salida}'.")