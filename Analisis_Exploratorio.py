import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans'})

#TITANIC

# Carpeta para guardar las imágenes individuales
carpeta_salida = "Exploración Titanic"
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar datasets
ruta_titanic = "titanic-Dataset.csv"
df_titanic = pd.read_csv(ruta_titanic)

# DISTRIBUCIÓN DE CLASES
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df_titanic, x='Survived', palette='Blues_d')
plt.title("1. Distribución de Clases", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Supervivencia (0 = No, 1 = Sí)")
plt.ylabel("Frecuencia")

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())} ({p.get_height()/len(df_titanic)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "1_distribucion_clases.png"), dpi=300)
plt.close()



# 2. DISTRIBUCIÓN NUMÉRICA: EDAD
plt.figure(figsize=(7, 5))
sns.histplot(df_titanic['Age'].dropna(), kde=True, color='teal', bins=30)
plt.title("2. Distribución Numérica: Edad", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Años")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "2_distribucion_edad.png"), dpi=300)
plt.close()



# 3. DISTRIBUCIÓN NUMÉRICA: TARIFA
plt.figure(figsize=(7, 5))
sns.histplot(df_titanic['Fare'], kde=True, color='indigo', bins=35)
plt.title("3. Distribución Numérica: Tarifa", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Tarifa ($)")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "3_distribucion_tarifa.png"), dpi=300)
plt.close()


# 4. BOXPLOT UNIVARIADO: OUTLIERS EN TARIFA
plt.figure(figsize=(7, 5))
sns.boxplot(x=df_titanic['Fare'], color='lightcoral')
plt.title("4. Boxplot Univariado: Detección de Outliers en Tarifa", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Tarifa ($)")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "4_boxplot_tarifa.png"), dpi=300)
plt.close()



# 5. BOXPLOT BIVARIADO: EDAD VS SUPERVIVENCIA
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_titanic, x='Survived', y='Age', palette='Set2')
plt.title("5. Boxplot Bivariado: Distribución de Edad según Supervivencia", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Survived (0 = No, 1 = Sí)")
plt.ylabel("Edad (años)")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "5_boxplot_edad_vs_supervivencia.png"), dpi=300)
plt.close()



# 6. FRECUENCIAS CATEGÓRICAS: CLASE Y SEXO
plt.figure(figsize=(7, 5))
sns.countplot(data=df_titanic, x='Pclass', hue='Sex', palette='crest')
plt.title("6. Frecuencias Categóricas: Clase de Pasaje agrupada por Sexo", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Clase (Pclass)")
plt.ylabel("Cantidad de pasajeros")
plt.legend(title="Sexo", loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "6_frecuencias_clase_sexo.png"), dpi=300)
plt.close()



# 7. RELACIÓN ATRIBUTO-OBJETIVO: SEXO VS SUPERVIVENCIA
plt.figure(figsize=(7, 5))
ax = sns.barplot(data=df_titanic, x='Sex', y='Survived', palette='mako', errorbar=None)
plt.title("7. Relación Atributo-Objetivo: Tasa de Supervivencia por Sexo", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Sexo")
plt.ylabel("Proporción que sobrevivió")

for p in ax.patches:
    ax.annotate(f'{p.get_height()*100:.1f}%',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "7_tasa_supervivencia_sexo.png"), dpi=300)
plt.close()



# 8. MATRIZ DE CORRELACIÓN
plt.figure(figsize=(7, 6))
cols_num_titanic = [c for c in ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] if c in df_titanic.columns]
corr_titanic = df_titanic[cols_num_titanic].corr()

sns.heatmap(corr_titanic, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8}, linewidths=0.5,
            xticklabels=cols_num_titanic, yticklabels=cols_num_titanic)

plt.title("8. Matriz de Correlación (Atributos Numéricos)", fontsize=13, pad=12, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "8_matriz_correlacion.png"), dpi=300)
plt.close()

print(f"IMAGENES DE TITANIC GUARDADAS'{carpeta_salida}'.")



#TELCO
# Carpeta para guardar las imágenes individuales
carpeta_salida = "Exploración Telco"
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar dataset de Telco
ruta_telco = "Telco-Customer-Churn-dataset.csv"
df_telco = pd.read_csv(ruta_telco)

#TotalCharges tiene espacios en blanco en clientes nuevos
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'].astype(str).str.strip(), errors='coerce')

#Mapeo (en español)
df_telco['Abandono_ES'] = df_telco['Churn'].map({'No': 'Permanece (No)', 'Yes': 'Abandona (Sí)'})
df_telco['Contrato_ES'] = df_telco['Contract'].map({
    'Month-to-month': 'Mensual',
    'One year': '1 Año',
    'Two year': '2 Años'
})

# 1. DISTRIBUCIÓN DE CLASES (DESBALANCE MODERADO)
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df_telco, x='Abandono_ES', palette=['#2b5c8f', '#d95f02'])
plt.title("1. Distribución de Clases", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Estado del Cliente")
plt.ylabel("Frecuencia")

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())} ({p.get_height()/len(df_telco)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "1_distribucion_clases.png"), dpi=300)
plt.close()


# 2. DISTRIBUCIÓN NUMÉRICA: ANTIGÜEDAD
plt.figure(figsize=(7, 5))
sns.histplot(df_telco['tenure'], kde=True, color='darkcyan', bins=30)
plt.title("2. Distribución Numérica: Antigüedad", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Meses de Permanencia")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "2_distribucion_antiguedad.png"), dpi=300)
plt.close()



# 3. DISTRIBUCIÓN NUMÉRICA: CARGOS MENSUALES (MONTHLYCHARGES)
plt.figure(figsize=(7, 5))
sns.histplot(df_telco['MonthlyCharges'], kde=True, color='chocolate', bins=30)
plt.title("3. Distribución Numérica: Cargos Mensuales", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Cargos Mensuales ($)")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "3_distribucion_cargos_mensuales.png"), dpi=300)
plt.close()



# 4. BOXPLOT UNIVARIADO: OUTLIERS EN CARGOS TOTALES (TOTALCHARGES)
plt.figure(figsize=(7, 5))
sns.boxplot(x=df_telco['TotalCharges'].dropna(), color='darkseagreen')
plt.title("4. Boxplot Univariado: Detección de Outliers en Cargos Totales", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Cargos Totales Acumulados ($)")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "4_boxplot_cargos_totales.png"), dpi=300)
plt.close()



# 5. BOXPLOT BIVARIADO: CARGOS MENSUALES VS ABANDONO
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_telco, x='Abandono_ES', y='MonthlyCharges', palette=['#74a9cf', '#fc9272'])
plt.title("5. Boxplot Bivariado: Cargos Mensuales según Abandono", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Estado del Cliente")
plt.ylabel("Cargo Mensual ($)")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "5_boxplot_cargos_vs_abandono.png"), dpi=300)
plt.close()



# 6. FRECUENCIAS CATEGÓRICAS: TIPO DE CONTRATO AGRUPADO POR ABANDONO
plt.figure(figsize=(7, 5))
sns.countplot(data=df_telco, x='Contrato_ES', hue='Abandono_ES', palette=['#2b5c8f', '#d95f02'])
plt.title("6. Frecuencias Categóricas: Tipo de Contrato agrupado por Abandono", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Modalidad de Contrato")
plt.ylabel("Cantidad de Clientes")
plt.legend(title="Estado", loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "6_frecuencias_contrato_abandono.png"), dpi=300)
plt.close()



# 7. RELACIÓN ATRIBUTO-OBJETIVO: TASA DE ABANDONO POR TIPO DE CONTRATO
plt.figure(figsize=(7, 5))
# Se calcula la tasa porcentual de abandono por modalidad
tasa_churn_contrato = df_telco.groupby('Contrato_ES')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()

ax = sns.barplot(data=tasa_churn_contrato, x='Contrato_ES', y='Churn', palette='mako', order=['Mensual', '1 Año', '2 Años'])
plt.title("7. Relación Atributo-Objetivo: Tasa de Abandono por Contrato", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Tipo de Contrato")
plt.ylabel("Proporción de Clientes que Abandonaron")

for p in ax.patches:
    ax.annotate(f'{p.get_height()*100:.1f}%',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "7_tasa_abandono_contrato.png"), dpi=300)
plt.close()



# 8. MATRIZ DE CORRELACIÓN (ATRIBUTOS CONTINUOS)
plt.figure(figsize=(7, 6))
cols_num_telco = [c for c in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'] if c in df_telco.columns]
corr_telco = df_telco[cols_num_telco].corr()

sns.heatmap(corr_telco, annot=True, cmap='mako', fmt=".2f", vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8}, linewidths=0.5,
            xticklabels=cols_num_telco, yticklabels=cols_num_telco)

plt.title("8. Matriz de Correlación (Atributos Continuos)", fontsize=13, pad=12, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "8_matriz_correlacion.png"), dpi=300)
plt.close()

print(f"IMAGENES DE TELCO GUARDADAS '{carpeta_salida}'.")