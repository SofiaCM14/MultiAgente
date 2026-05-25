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
    Calcula estadísticas descriptivas básicas y genera un histograma 
    junto con una matriz de correlación.
    """
    # Estadísticas descriptivas de variables numéricas
    estadisticas = df.describe()
    
    # Conteo de valores nulos por columna
    valores_nulos = df.isnull().sum().to_frame(name="Total Nulos")
    
    # 1. Histograma de la distribución de ventas
    fig_hist, ax_hist = plt.subplots(figsize=(5, 3.5))
    if 'Ventas' in df.columns:
        sns.histplot(df['Ventas'].dropna(), kde=True, ax=ax_hist, color='royalblue')
    ax_hist.set_title('Distribución de Ventas', fontsize=10)
    plt.tight_layout()
    
    # 2. Matriz de correlación
    fig_corr, ax_corr = plt.subplots(figsize=(5, 3.5))
    df_numerico = df.select_dtypes(include=[np.number])
    sns.heatmap(df_numerico.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, cbar=False)
    ax_corr.set_title('Matriz de Correlación', fontsize=10)
    plt.tight_layout()
    
    return estadisticas, valores_nulos, fig_hist, fig_corr


# --- AGENTE 2: LIMPIEZA DE DATOS ---
def agente_limpieza(df):
    """
    Elimina duplicados, imputa valores nulos en columnas críticas 
    y aplica codificación categórica inteligente evitando explosión de columnas.
    """
    df_limpio = df.copy()
    
    # 1. Eliminar filas duplicadas
    df_limpio = df_limpio.drop_duplicates()
    
    # 2. Imputación de valores nulos
    for col in df_limpio.columns:
        if df_limpio[col].isnull().sum() > 0:
            if df_limpio[col].dtype in [np.float64, np.int64]:
                # Imputar numéricas con la mediana
                df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
            else:
                # Imputar categóricas con la moda
                df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode()[0] if not df_limpio[col].mode().empty else "Desconocido")
                
    # 3. Codificación inteligente de variables categóricas
    columnas_categoricas = []
    columnas_alta_cardinalidad = []
    
    # Filtramos columnas como ID_Transaccion o Fecha que tienen demasiados valores únicos
    for col in df_limpio.select_dtypes(include=['object', 'category']).columns:
        if df_limpio[col].nunique() <= 50:
            columnas_categoricas.append(col) # Entran Categoria y Prenda
        else:
            columnas_alta_cardinalidad.append(col) # Se descartan IDs y Fechas para que no explote la RAM
            
    # Eliminamos las columnas de texto masivo que no aportan al patrón matemático
    df_limpio = df_limpio.drop(columns=columnas_alta_cardinalidad)
    
    # Aplicamos One-Hot Encoding solo a las categorías estables
    df_codificado = pd.get_dummies(df_limpio, columns=columnas_categoricas, drop_first=True)
    
    return df_codificado


# --- AGENTE 3: MODELO PREDICTIVO ---
def agente_prediccion(df):
    """
    Entrena Regresión Lineal y Árbol de Decisión, compara su desempeño 
    mediante RMSE y genera estimaciones interpretables.
    """
    if 'Ventas' not in df.columns:
        raise ValueError("El dataset debe contener una columna objetivo de ventas.")
        
    # X son las características. Nos aseguramos de entrenar SOLO con columnas numéricas
    X = df.drop(columns=['Ventas'])
    X = X.select_dtypes(include=[np.number])
    y = df['Ventas']
    
    # División de datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Modelo de Regresión Lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, preds_lr))
    
    # 2. Modelo de Árbol de Decisión
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    preds_dt = dt.predict(X_test)
    rmse_dt = np.sqrt(mean_squared_error(y_test, preds_dt))
    
    # 3. Selección del mejor modelo (menor RMSE)
    if rmse_lr < rmse_dt:
        mejor_nombre = "Regresión Lineal"
        rmse_mejor = rmse_lr
        modelo_ganador = lr
    else:
        mejor_nombre = "Árbol de Decisión"
        rmse_mejor = rmse_dt
        modelo_ganador = dt
        
    # Generar predicciones interpretables sobre el dataset completo
    df_resultados = df.copy()
    df_resultados['Ventas_Predichas'] = modelo_ganador.predict(X)
    
    # Cambiamos los nombres para la tabla final para que sea súper estético
    df_resultados = df_resultados.rename(columns={'Ventas': 'Ventas Reales', 'Ventas_Predichas': 'Ventas Predichas'})
    
    return mejor_nombre, rmse_lr, rmse_dt, rmse_mejor, df_resultados[['Ventas Reales', 'Ventas Predichas']]