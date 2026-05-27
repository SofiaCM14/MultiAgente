import pandas as pd
import numpy as np
import matplotlib
# LE DECIMOS A LA PC QUE NO ABRA VENTANAS FLOTANTES DE GRÁFICOS.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
import os
from typing import Tuple, Dict, Any

# AGENTE 1: EL ANALISTA (EDA - Análisis Exploratorio de Datos)
class AgenteEDA:
    def __init__(self):
        self.nombre = "Agente_EDA"

    def ejecutar(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str, str]:
        print(f"[{self.nombre}]: Hola! Estoy revisando el archivo por primera vez...")
        
        if df.empty:
            raise ValueError("El archivo de datos está completamente vacío.")
            
        col_ventas = None
        for col in df.columns:
            if 'venta' in col.lower() or 'sales' in col.lower() or 'total' in col.lower():
                col_ventas = col
                break
        
        if not col_ventas:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                col_ventas = num_cols[-1]
            else:
                raise KeyError("No encontré ninguna columna numérica que represente las 'Ventas'.")

        estadisticas = df.describe(include='all')
        valores_nulos = df.isnull().sum()
        
        os.makedirs('static', exist_ok=True)
        path_hist = 'static/histograma_ventas.png'
        path_corr = 'static/matriz_correlacion.png'

        # Histograma
        plt.figure(figsize=(7, 4.5))
        sns.histplot(df[col_ventas], kde=True, color='blue', bins=30)
        plt.title(f'Distribución de las Ventas: {col_ventas}', fontsize=12, fontweight='bold')
        plt.xlabel('Dinero ($)')
        plt.ylabel('Cuántas veces ocurrió')
        plt.tight_layout()
        plt.savefig(path_hist, dpi=150)
        plt.close()
        
        # Matriz de Correlación
        plt.figure(figsize=(7, 4.5))
        columnas_num = df.select_dtypes(include=[np.number])
        if columnas_num.shape[1] > 1:
            sns.heatmap(columnas_num.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        else:
            sns.heatmap(columnas_num.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matriz de Correlación (Qué variables se parecen)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path_corr, dpi=150)
        plt.close()
        
        print(f"[{self.nombre}]: Terminé! Le paso mis resultados al Orquestador.")
        return estadisticas, valores_nulos, path_hist, path_corr

# AGENTE 2: EL LIMPIADOR (Limpieza)
class AgenteLimpieza:
    def __init__(self):
        self.nombre = "Agente_Limpieza"

    def ejecutar(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]: #Entrada y salida del agente
        print(f"[{self.nombre}]: Iniciando proceso de limpieza profunda y restauración...")
        df_limpio = df.copy() 
        
        # Conteo inicial de nulos antes de limpiar
        nulos_totales_antes = int(df_limpio.isnull().sum().sum())
        
        # Filtrado de columnas irrelevantes
        columnas_a_quitar = ['Fecha', 'ID_Transaccion', 'Prenda', 'fecha', 'id_transaccion', 'prenda']
        for col in columnas_a_quitar:
            if col in df_limpio.columns:
                df_limpio = df_limpio.drop(columns=[col])

        # Identificación y eliminación de filas duplicadas
        duplicados_antes = int(df_limpio.duplicated().sum())
        df_limpio = df_limpio.drop_duplicates()
        
        # Imputación científica de casillas vacías (Evita errores NaN en Agente 3)
        for col in df_limpio.columns:
            if df_limpio[col].isnull().sum() > 0:
                if df_limpio[col].dtype in [np.float64, np.int64]:
                    # Rellenamos números con la Mediana estadistica
                    df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
                else:
                    # Rellenamos categorías de texto con la Moda (lo que más se repite)
                    moda = df_limpio[col].mode()
                    df_limpio[col] = df_limpio[col].fillna(moda[0] if not moda.empty else "Desconocido")

        # Guardamos mapeo de texto para la interpretación humana posterior
        categorias_origen = {}
        columnas_cat = df_limpio.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in columnas_cat:
            categorias_origen[col] = df_limpio[col].unique().tolist()
            
        # One-Hot Encoding (Conversión matemática a binario)
        if len(columnas_cat) > 0:
            df_limpio = pd.get_dummies(df_limpio, columns=columnas_cat, drop_first=True, dtype=int)
            
        metadatos_limpieza = {
            "duplicados_eliminados": duplicados_antes,
            "nulos_reparados": nulos_totales_antes,
            "columnas_categorizadas": columnas_cat,
            "mapeo_categorias": categorias_origen
        }
        
        print(f"[{self.nombre}]: Pipeline de limpieza completado con éxito.")
        return df_limpio, metadatos_limpieza


# AGENTE 3: EL MATEMÁTICO (Modelo Predictivo) - CORREGIDO
# Su trabajo es entrenar dos inteligencias artificiales diferentes, comparar
# cuál comete menos errores usando RMSE y dar predicciones de ventas.
class AgentePrediccion:
    def __init__(self):
        self.nombre = "Agente_Prediccion"

    def ejecutar(self, df_limpio: pd.DataFrame, metadatos: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[{self.nombre}]: Iniciando el entrenamiento de los modelos...")
        
        # Buscamos otra vez cuál es la columna de ventas en el archivo ya limpio
        col_ventas = None
        for col in df_limpio.columns:
            if 'venta' in col.lower() or 'sales' in col.lower() or 'total' in col.lower():
                col_ventas = col
                break
        if not col_ventas:
            num_cols = df_limpio.select_dtypes(include=[np.number]).columns
            col_ventas = num_cols[-1]

        # SEPARACIÓN DE DATOS:
        # X = Todo lo que usamos para predecir (Precio, Cantidad, Descuento, etc.)
        # y = Lo que queremos adivinar (Las Ventas)
        X = df_limpio.drop(columns=[col_ventas])
        y = df_limpio[col_ventas]
        
        # Dividimos los datos usando un 80% para entrenar los modelos
        # y dejamos un 20% guardado en una caja fuerte (X_test, y_test) para ponerlos a prueba.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # MODELO A: REGRESIÓN LINEAL
        modelo_lr = LinearRegression()
        modelo_lr.fit(X_train, y_train) # Entrenar / Aprender
        pred_lr = modelo_lr.predict(X_test) # Hacer examen de prueba
        rmse_lr = root_mean_squared_error(y_test, pred_lr) 
        
        # MODELO B: ÁRBOL DE DECISIÓN
        modelo_dt = DecisionTreeRegressor(random_state=42, max_depth=5)
        modelo_dt.fit(X_train, y_train) # Entrenar / Aprender
        pred_dt = modelo_dt.predict(X_test) # Hacer examen de prueba
        rmse_dt = root_mean_squared_error(y_test, pred_dt) # Sacar su error RMSE 
        
        # SELECCIÓN DEL GANADOR: 
        if rmse_lr < rmse_dt:
            mejor_nombre = "Regresión Lineal"
            mejor_modelo = modelo_lr
            mejor_rmse = rmse_lr
        else:
            mejor_nombre = "Árbol de Decisión"
            mejor_modelo = modelo_dt
            mejor_rmse = rmse_dt
            
        # PREDICCIONES INTERPRETABLES (VENTAS ESTIMADAS POR CATEGORÍA) 
        predicciones_humanas = {}
        columnas_cat_originales = metadatos.get("columnas_categorizadas", [])
        
        if columnas_cat_originales:
            col_principal = columnas_cat_originales[0] # Ej. 'Categoria'
            valores_cat = metadatos["mapeo_categorias"].get(col_principal, [])
            
            # Sacamos un renglón promedio con características numéricas genéricas del negocio
            fila_promedio = X.mean().to_frame().T
            
            # 🌟 CORRECCIÓN CLAVE: Identificamos cuáles columnas del set X son dummies de categoría
            columnas_dummies = [c for c in X.columns if c.startswith(f"{col_principal}_")]
            
            # Simulamos el comportamiento para cada categoría individual
            for cat in valores_cat:
                fila_simulada = fila_promedio.copy()
                
                # Apagamos absolutamente todas las dummies (las ponemos en 0) para limpiar el renglón promedio
                for col_d in columnas_dummies:
                    fila_simulada[col_d] = 0
                
                # Intentamos encender el interruptor de la categoría actual
                col_dummy = f"{col_principal}_{cat}"
                if col_dummy in fila_simulada.columns:
                    fila_simulada[col_dummy] = 1 # Si existe, la encendemos con 1
                # Nota: Si no existe en las columnas, significa que es la categoría base (el primer drop)
                # y se predice correctamente dejando todas las demás dummies en 0.
                
                # Le pedimos al modelo ganador que nos diga la venta estimada en pesos
                ventas_estimadas = mejor_modelo.predict(fila_simulada)[0]
                predicciones_humanas[cat] = round(max(0.0, ventas_estimadas), 2)
        else:
            # Si el archivo no tenía texto, damos estimados generales usando percentiles
            predicciones_humanas["Ventas Mínimas Estimadas"] = round(max(0.0, float(np.percentile(pred_dt, 25))), 2)
            predicciones_humanas["Ventas Promedio Estimadas"] = round(float(np.mean(pred_dt)), 2)
            predicciones_humanas["Ventas Máximas Estimadas"] = round(float(np.percentile(pred_dt, 75)), 2)

        print(f"[{self.nombre}]: Modelado listo! El modelo ganador fue: {mejor_nombre}")
        return {
            "rmse_regresion_lineal": round(rmse_lr, 4),
            "rmse_arbol_decision": round(rmse_dt, 4),
            "modelo_seleccionado": mejor_nombre,
            "rmse_mejor": round(mejor_rmse, 4),
            "predicciones_categoria": predicciones_humanas
        }

# =========================================================================
# AGENTE ORQUESTADOR: COORDINA EL PIPELINE
# =========================================================================
class AgenteOrquestador:
    def __init__(self):
        self.nombre = "Orquestador_Central"
        self.agente_eda = AgenteEDA()
        self.agente_limpieza = AgenteLimpieza()
        self.agente_prediccion = AgentePrediccion()
        
    def procesar_pipeline(self, ruta_archivo: str) -> Dict[str, Any]:
        print(f"\n================ [{self.nombre}]: INICIANDO PIPELINE ================")
        try:
            if ruta_archivo.endswith('.csv'):
                df_inicial = pd.read_csv(ruta_archivo)
            elif ruta_archivo.endswith(('.xlsx', '.xls')):
                df_inicial = pd.read_excel(ruta_archivo)
            else:
                raise ValueError("Formato no soportado. Sube .csv o .xlsx")
                
            estadisticas, nulos, path_hist, path_corr = self.agente_eda.ejecutar(df_inicial)
            df_limpio, metadatos_limpieza = self.agente_limpieza.ejecutar(df_inicial)
            resultados_modelos = self.agente_prediccion.ejecutar(df_limpio, metadatos_limpieza)
            
            return {
                "estado": "Éxito",
                "eda_stats": estadisticas,
                "eda_nulos": nulos,
                "grafico_histograma": path_hist,
                "grafico_correlacion": path_corr,
                "limpieza_meta": metadatos_limpieza,
                "modelos": resultados_modelos
            }
        except Exception as e:
            print(f"❌ [{self.nombre}]: Problema en el flujo: {str(e)}")
            return {"estado": "Error", "mensaje": str(e)}