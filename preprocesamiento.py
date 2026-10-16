import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. PREPROCESAMIENTO PARA TELCO 
def preprocesar_telco(filepath):
    df = pd.read_csv(filepath)
    
    # A. Limpieza de TotalCharges (espacios en blanco a numérico)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    
    # Imputar 0.0 en TotalCharges para clientes con tenure = 0
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0.0
    
    # B. Separación de características y variable objetivo
    # Si la columna es 'Abandono_ES' o 'Churn', ajustamos
    target_col = 'Churn' if 'Churn' in df.columns else 'Abandono_ES'
    
    X = df.drop(columns=['customerID', target_col], errors='ignore')
    y = df[target_col].map({'Yes': 1, 'No': 0, 'Si': 1, 'Sí': 1}) if df[target_col].dtype == 'object' else df[target_col]
    
    # C. Transformación Logarítmica para asimetría en TotalCharges
    X['TotalCharges_log'] = np.log1p(X['TotalCharges'])
    X = X.drop(columns=['TotalCharges'])
    
    # Identificar columnas por tipo
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges_log', 'SeniorCitizen']
    cat_cols = [col for col in X.columns if col not in num_cols]
    
    # Pipelines de transformación
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Recuperar nombres de columnas
    cat_encoder = preprocessor.named_transformers_['cat']['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(cat_feature_names)
    
    X_df = pd.DataFrame(X_processed, columns=all_feature_names)
    
    return X_df, y

# 2. PREPROCESAMIENTO PARA TITANIC
def preprocesar_titanic(filepath):
    df = pd.read_csv(filepath)
    
    # A. Separación de características y variable objetivo
    X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], errors='ignore')
    y = df['Survived']
    
    # B. Transformación Logarítmica para asimetría en Fare
    X['Fare_log'] = np.log1p(X['Fare'])
    X = X.drop(columns=['Fare'])
    
    # Identificar columnas
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare_log']
    cat_cols = ['Pclass', 'Sex', 'Embarked']
    
    # Pipelines de transformación
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Recuperar nombres de columnas
    cat_encoder = preprocessor.named_transformers_['cat']['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(cat_feature_names)
    
    X_df = pd.DataFrame(X_processed, columns=all_feature_names)
    
    return X_df, y

# EJECUCIÓN PRUEBA
if __name__ == "__main__":
    print("Preprocesamiento listo para ejecutarse sobre los datasets.")