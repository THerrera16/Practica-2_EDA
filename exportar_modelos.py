import joblib
from sklearn.linear_model import LogisticRegression

from preprocesamiento import preprocesar_telco, preprocesar_titanic
from balanceo_y_split import preparar_datos_balanceados

def guardar_modelos():
    print(" MODELOS ENTRENANDOS Y GUARDADOS")
    
    # 1. Modelo Final Telco
    X_telco, y_telco = preprocesar_telco("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_tr_t, X_te_t, y_tr_t, y_te_t = preparar_datos_balanceados(X_telco, y_telco)
    
    y_tr_t_num = y_tr_t.map({'Yes': 1, 'No': 0, 1: 1, 0: 0}) if hasattr(y_tr_t, 'map') else y_tr_t
    
    model_telco = LogisticRegression(C=1, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
    model_telco.fit(X_tr_t, y_tr_t_num)
    
    joblib.dump(model_telco, 'modelo_final_telco.pkl')
    print("Modelo de Telco guardado como 'modelo_final_telco.pkl'")
    
    # 2. Modelo Final Titanic
    X_titanic, y_titanic = preprocesar_titanic("datasets/Titanic-Dataset.csv")
    X_tr_ti, X_te_ti, y_tr_ti, y_te_ti = preparar_datos_balanceados(X_titanic, y_titanic)
    
    model_titanic = LogisticRegression(C=1, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
    model_titanic.fit(X_tr_ti, y_tr_ti)
    
    joblib.dump(model_titanic, 'modelo_final_titanic.pkl')
    print("Modelo de Titanic guardado como 'modelo_final_titanic.pkl'")

if __name__ == "__main__":
    guardar_modelos()