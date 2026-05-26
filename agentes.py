import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# --- AGENTE 1: EDA ---
def agente_eda(df):
    """
    Calcula estadísticas descriptivas básicas y genera gráficos obligatorios.
    Soporta cualquier estructura de datos.
    """
    estadisticas = df.describe()
    valores_nulos = df.isnull().sum().to_frame(name="Total Nulos")
    
    # 1. Histograma de la distribución de ventas (si existe la columna)
    fig_hist, ax_hist = plt.subplots(figsize=(5, 3.5))
    if 'Ventas' in df.columns:
        sns.histplot(df['Ventas'].dropna(), kde=True, ax=ax_hist, color='royalblue')
        ax_hist.set_title('Distribución de Ventas', fontsize=10)
    else:
        ax_hist.text(0.5, 0.5, 'Columna "Ventas" no encontrada', ha='center', va='center')
    plt.tight_layout()
    
    # 2. Matriz de correlación dinámica (solo toma las numéricas que encuentre)
    fig_corr, ax_corr = plt.subplots(figsize=(5, 3.5))
    df_numerico = df.select_dtypes(include=[np.number])
    if df_numerico.shape[1] > 1:
        sns.heatmap(df_numerico.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, cbar=False)
        ax_corr.set_title('Matriz de Correlación', fontsize=10)
    else:
        ax_corr.text(0.5, 0.5, 'Faltan variables numéricas para correlación', ha='center', va='center')
    plt.tight_layout()
    
    return estadisticas, valores_nulos, fig_hist, fig_corr


# --- AGENTE 2: LIMPIEZA DE DATOS ---
def agente_limpieza(df):
    """
    Elimina duplicados, imputa nulos dinámicamente según el tipo de datos
    y dummifica variables categóricas estables de cualquier archivo.
    """
    df_limpio = df.copy()
    
    # 1. Eliminar duplicados
    df_limpio = df_limpio.drop_duplicates()
    
    # 2. Imputación de nulos dinámica
    for col in df_limpio.columns:
        if df_limpio[col].isnull().sum() > 0:
            if df_limpio[col].dtype in [np.float64, np.int64]:
                df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
            else:
                df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode()[0] if not df_limpio[col].mode().empty else "Desconocido")
                
    # 3. Codificación categórica inteligente
    columnas_categoricas = []
    columnas_a_eliminar = []
    
    for col in df_limpio.select_dtypes(include=['object', 'category']).columns:
        # Si la columna tiene demasiados valores únicos (ej. IDs, folios, fechas, nombres), se descarta
        if df_limpio[col].nunique() <= 50:
            columnas_categoricas.append(col)
        else:
            columnas_a_eliminar.append(col)
            
    df_limpio = df_limpio.drop(columns=columnas_a_eliminar)
    
    # Aplicar One-Hot Encoding solo a las categóricas válidas
    if columnas_categoricas:
        df_codificado = pd.get_dummies(df_limpio, columns=columnas_categoricas, drop_first=True)
    else:
        df_codificado = df_limpio
        
    return df_codificado


# --- AGENTE 3: MODELO PREDICTIVO ---
def agente_prediccion(df):
    """
    Entrena y compara Regresión Lineal y Árbol de Decisión usando cualquier columna numérica remanente.
    """
    if 'Ventas' not in df.columns:
        raise ValueError("El dataset procesado no contiene la columna objetivo 'Ventas'.")
        
    # Filtrar solo características numéricas para evitar errores en Sklearn
    X = df.drop(columns=['Ventas']).select_dtypes(include=[np.number])
    y = df['Ventas']
    
    # Validación por si el archivo no tiene más columnas numéricas con qué predecir
    if X.shape[1] == 0:
        # Si no hay variables, creamos una columna de índice temporal para que no truene el modelo
        X['Indice_Temporal'] = np.arange(len(df))
        
    # División 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Regresión Lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, preds_lr))
    
    # 2. Árbol de Decisión
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    preds_dt = dt.predict(X_test)
    rmse_dt = np.sqrt(mean_squared_error(y_test, preds_dt))
    
    # 3. Selección del mejor
    if rmse_lr < rmse_dt:
        mejor_nombre = "Regresión Lineal"
        rmse_mejor = rmse_lr
        modelo_ganador = lr
    else:
        mejor_nombre = "Árbol de Decisión"
        rmse_mejor = rmse_dt
        modelo_ganador = dt
        
    # Generar resultados interpretables
    df_resultados = pd.DataFrame({
        'Ventas Reales': y,
        'Ventas Predichas': modelo_ganador.predict(X)
    })
    
    return mejor_nombre, rmse_lr, rmse_dt, rmse_mejor, df_resultados