import pandas as pd
import numpy as np
import matplotlib
# LE DECIMOS A LA PC QUE NO ABRA VENTANAS FLOTANTES DE GRÁFICOS.
# Si no ponemos esto, la página web de Streamlit puede trabarse o congelarse.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
import os
from typing import Tuple, Dict, Any

# =========================================================================
# AGENTE 1: EL ANALISTA (EDA - Análisis Exploratorio de Datos)
# Su único trabajo es revisar el archivo que subió el usuario, calcular
# números básicos (como promedios) y hacer dos dibujos/gráficas.
# =========================================================================
class AgenteEDA:
    def __init__(self):
        # Le ponemos un nombre al agente para identificarlo en la terminal
        self.nombre = "Agente_EDA"

    def ejecutar(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str, str]:
        print(f"[{self.nombre}]: Hola! Estoy revisando el archivo por primera vez...")
        
        # Aquí validamos que si el archivo no tiene renglones, el sistema se detiene y avisa.
        if df.empty:
            raise ValueError("El archivo de datos está completamente vacío.")
            
        # BUSCADOR AUTOMÁTICO DE LA COLUMNA DE VENTAS:
        # Los archivos de retail pueden llamar a las ventas como 'Ventas', 'Sales' o 'Total'.
        # Este pedazo de código busca en los títulos del archivo alguna palabra similar para no fallar.
        col_ventas = None
        for col in df.columns:
            if 'venta' in col.lower() or 'sales' in col.lower() or 'total' in col.lower():
                col_ventas = col
                break
        
        # Si de plano no encontró ninguna palabra clave, agarra la última columna numérica por defecto.
        if not col_ventas:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                col_ventas = num_cols[-1]
            else:
                raise KeyError("No encontré ninguna columna numérica que represente las 'Ventas'.")

        # PASO 1: SACAR LAS ESTADÍSTICAS BÁSICAS OBLIGATORIAS
        # .describe(include='all') calcula la media, mediana, máximos y mínimos de todo el archivo.
        estadisticas = df.describe(include='all')
        # .isnull().sum() cuenta cuántas casillas en blanco/vacías hay en cada columna.
        valores_nulos = df.isnull().sum()
        
        # Creamos una carpeta llamada 'static' en tu computadora para guardar las fotos de los gráficos
        os.makedirs('static', exist_ok=True)
        path_hist = 'static/histograma_ventas.png'
        path_corr = 'static/matriz_correlacion.png'

        # PASO 2: DIBUJAR EL HISTOGRAMA (DIBUJO 1)
        # Muestra cómo se distribuyen las ventas (si la gente compra poquito, mucho o normal).
        plt.figure(figsize=(7, 4.5))
        sns.histplot(df[col_ventas], kde=True, color='blue', bins=30)
        plt.title(f'Distribución de las Ventas: {col_ventas}', fontsize=12, fontweight='bold')
        plt.xlabel('Dinero ($)')
        plt.ylabel('Cuántas veces ocurrió')
        plt.tight_layout()
        plt.savefig(path_hist, dpi=150) # Guarda el dibujo como foto
        plt.close() # Cierra el lienzo para que no gaste memoria de la PC
        
        # PASO 3: DIBUJAR LA MATRIZ DE CORRELACIÓN (DIBUJO 2)
        # Es un mapa de calor con números que nos dice qué columnas están amarradas o relacionadas entre sí.
        plt.figure(figsize=(7, 4.5))
        columnas_num = df.select_dtypes(include=[np.number]) # Solo usamos columnas con números
        if columnas_num.shape[1] > 1:
            sns.heatmap(columnas_num.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        else:
            sns.heatmap(columnas_num.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matriz de Correlación (Qué variables se parecen)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path_corr, dpi=150) # Guarda el mapa como foto
        plt.close()
        
        print(f"[{self.nombre}]: Terminé! Le paso mis resultados al Orquestador.")
        # Devolvemos las tablas numéricas y las rutas de las fotos para que Streamlit las pinte en pantalla.
        return estadisticas, valores_nulos, path_hist, path_corr


# =========================================================================
# AGENTE 2: EL LIMPIADOR (Preprocesamiento y Limpieza de Datos)
# Su único trabajo es reparar el archivo: borrar renglones repetidos,
# rellenar huecos vacíos y convertir palabras a números para la IA.
# =========================================================================
class AgenteLimpieza:
    def __init__(self):
        self.nombre = "Agente_Limpieza"

    def ejecutar(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        print(f"[{self.nombre}]: Recibí los datos dañados. Empezando a limpiar...")
        df_limpio = df.copy() # Hacemos una copia para no arruinar el archivo original
        
        # 🌟 PASO 0: FILTRADO DE COLUMNAS INÚTILES PARA LA IA (¡ESTA ES LA MODIFICACIÓN!)
        # Quitamos la Fecha y el ID de Transacción porque cada renglón tiene uno distinto y no ayudan a predecir dinero.
        # Quitamos también 'Prenda' porque al ser nombres específicos de ropa (como 'Jeans Cargo') generaría miles de columnas dummies.
        # Nos quedamos con 'Categoria' que agrupa todo de forma ordenada (como 'Partes Inferiores').
        columnas_a_quitar = ['Fecha', 'ID_Transaccion', 'Prenda', 'fecha', 'id_transaccion', 'prenda']
        for col in columnas_a_quitar:
            if col in df_limpio.columns:
                df_limpio = df_limpio.drop(columns=[col])

        # PASO 1: ELIMINAR RENGLONES DUPLICADOS (REPETIDOS)
        duplicados_antes = df_limpio.duplicated().sum() # Cuenta cuántos repetidos hay
        df_limpio = df_limpio.drop_duplicates() # Los borra permanentemente
        
        # PASO 2: RELLENAR CASILLAS EN BLANCO (IMPUTACIÓN)
        # ¿Con qué criterio rellenaron los nulos? Si la columna es un número, usamos la 'MEDIANA' porque no se altera con valores raros.
        # Si la columna es texto (categoría), usamos la 'MODA' (el texto que más se repite).
        for col in df_limpio.columns:
            if df_limpio[col].isnull().sum() > 0:
                if df_limpio[col].dtype in [np.float64, np.int64]:
                    df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median()) # Rellena números
                else:
                    moda = df_limpio[col].mode()
                    df_limpio[col] = df_limpio[col].fillna(moda[0] if not moda.empty else "Desconocido") # Rellena textos
        
        # Guardamos una lista de las categorías originales (ej. Ropa, Electrónica) antes de cambiarlas.
        # Esto sirve para que el Agente 3 pueda dar respuestas que un humano entienda más adelante.
        categorias_origen = {}
        columnas_cat = df_limpio.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in columnas_cat:
            categorias_origen[col] = df_limpio[col].unique().tolist()
            
        # PASO 3: CONVERTIR PALABRAS A NÚMEROS (ONE-HOT ENCODING)
        # Los modelos matemáticos no entienden la palabra "Ropa" o "Zapatos".
        # pd.get_dummies convierte esas palabras en columnas de unos (1) y ceros (0).
        if len(columnas_cat) > 0:
            df_limpio = pd.get_dummies(df_limpio, columns=columnas_cat, drop_first=True, dtype=int)
            
        # Guardamos un reporte detallado de qué limpiamos para pasárselo al reporte técnico.
        metadatos_limpieza = {
            "duplicados_eliminados": duplicados_antes,
            "columnas_categorizadas": columnas_cat,
            "mapeo_categorias": categorias_origen
        }
        
        print(f"[{self.nombre}]: Datos limpios y listos para los modelos matemáticos.")
        return df_limpio, metadatos_limpieza


# =========================================================================
# AGENTE 3: EL MATEMÁTICO (Modelo Predictivo)
# Su trabajo es entrenar dos inteligencias artificiales diferentes, comparar
# cuál comete menos errores usando RMSE y dar predicciones de ventas.
# =========================================================================
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
        # X = Todo lo que usamos para predecir (Precio, Inventario, Categoría, etc.)
        # y = Lo que queremos adivinar (Las Ventas)
        X = df_limpio.drop(columns=[col_ventas])
        y = df_limpio[col_ventas]
        
        # EXPLICACIÓN AL DOC: Dividimos los datos usando un 80% para entrenar los modelos
        # y dejamos un 20% guardado en una caja fuerte (X_test, y_test) para ponerlos a prueba.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # MODELO A: REGRESIÓN LINEAL
        # Es un modelo que asume que los datos crecen o decrecen formando una línea recta.
        modelo_lr = LinearRegression()
        modelo_lr.fit(X_train, y_train) # Entrenar / Aprender
        pred_lr = modelo_lr.predict(X_test) # Hacer examen de prueba
        # RMSE: Es el promedio de qué tan lejos se quedó el modelo de la respuesta real (El error).
        rmse_lr = root_mean_squared_error(y_test, pred_lr) 
        
        # MODELO B: ÁRBOL DE DECISIÓN
        # Es un modelo que hace preguntas lógicas tipo árbol ("¿El precio es mayor a $100? Sí/No").
        # Ponemos max_depth=5 para que el árbol no crezca infinitamente y no memorice las respuestas (sobreajuste).
        modelo_dt = DecisionTreeRegressor(random_state=42, max_depth=5)
        modelo_dt.fit(X_train, y_train) # Entrenar / Aprender
        pred_dt = modelo_dt.predict(X_test) # Hacer examen de prueba
        rmse_dt = root_mean_squared_error(y_test, pred_dt) # Sacar su error RMSE 
        
        # SELECCIÓN DEL GANADOR: 
        # El modelo que tenga el número RMSE MÁS CHICO es el ganador, porque se equivocó menos.
        if rmse_lr < rmse_dt:
            mejor_nombre = "Regresión Lineal"
            mejor_modelo = modelo_lr
            mejor_rmse = rmse_lr
        else:
            mejor_nombre = "Árbol de Decisión"
            mejor_modelo = modelo_dt
            mejor_rmse = rmse_dt
            
        # PREDICCIONES INTERPRETABLES (VENTAS ESTIMADAS POR CATEGORÍA) 
        # Como el profesor quiere ver predicciones lógicas humanas, aquí reconstruimos
        # las categorías originales para calcular cuánto venderá cada una en promedio.
        predicciones_humanas = {}
        columnas_cat_originales = metadatos.get("columnas_categorizadas", [])
        
        if columnas_cat_originales:
            col_principal = columnas_cat_originales[0] # Ej. 'Categoria'
            valores_cat = metadatos["mapeo_categorias"].get(col_principal, [])
            
            # Sacamos un renglón promedio con características genéricas del comercio
            fila_promedio = X.mean().to_frame().T
            
            # Simulamos el comportamiento para cada categoría individual
            for cat in valores_cat:
                fila_simulada = fila_promedio.copy()
                col_dummy = f"{col_principal}_{cat}"
                if col_dummy in fila_simulada.columns:
                    fila_simulada[col_dummy] = 1 # Encendemos el interruptor de esa categoría
                
                # Le pedimos al modelo ganador que nos diga la venta estimada en pesos
                ventas_estimadas = mejor_modelo.predict(fila_simulada)[0]
                # max(0.0, ...) evita que la IA de números negativos absurdos (no puedes vender -$5 pesos)
                predicciones_humanas[cat] = round(max(0.0, ventas_estimadas), 2)
        else:
            # Si el archivo no tenía texto, damos estimados generales usando percentiles
            predicciones_humanas["Ventas Mínimas Estimadas"] = round(max(0.0, float(np.percentile(pred_dt, 25))), 2)
            predicciones_humanas["Ventas Promedio Estimadas"] = round(float(np.mean(pred_dt)), 2)
            predicciones_humanas["Ventas Máximas Estimadas"] = round(float(np.percentile(pred_dt, 75))), 2

        print(f"[{self.nombre}]: Modelado listo! El modelo ganador fue: {mejor_nombre}")
        return {
            "rmse_regresion_lineal": round(rmse_lr, 4),
            "rmse_arbol_decision": round(rmse_dt, 4),
            "modelo_seleccionado": mejor_nombre,
            "rmse_mejor": round(mejor_rmse, 4),
            "predicciones_categoria": predicciones_humanas
        }


# =========================================================================
# AGENTE ORQUESTADOR: EL JEFE DE JEFES (Coordinador del Sistema Multiagente)
# Su único trabajo es controlar el tiempo y el orden del flujo:
# Llama al Agente 1, toma su salida, se la pasa al Agente 2, luego al Agente 3
# y al final reúne los resultados para entregárselos a la página web.
# =========================================================================
class AgenteOrquestador:
    def __init__(self):
        self.nombre = "Orquestador_Central"
        # El Orquestador tiene contratados/guardados a los 3 agentes especializados 
        self.agente_eda = AgenteEDA()
        self.agente_limpieza = AgenteLimpieza()
        self.agente_prediccion = AgentePrediccion()
        
    def procesar_pipeline(self, ruta_archivo: str) -> Dict[str, Any]:
        print(f"\n================ [{self.nombre}]: INICIANDO CONTROL DEL PIPELINE ================")
        
        try:
            # 1. Leer el archivo según su formato de extensión (.csv o .xlsx) 
            if ruta_archivo.endswith('.csv'):
                df_inicial = pd.read_csv(ruta_archivo)
            elif ruta_archivo.endswith(('.xlsx', '.xls')):
                df_inicial = pd.read_excel(ruta_archivo)
            else:
                raise ValueError("Formato no soportado. Por favor sube un .csv o un .xlsx")
                
            # FASE 1 SECUENCIAL: Arranca el Agente 1 (EDA) 
            estadisticas, nulos, path_hist, path_corr = self.agente_eda.ejecutar(df_inicial)
            
            # FASE 2 SECUENCIAL: Los datos iniciales van al Agente 2 para limpiarse 
            df_limpio, metadatos_limpieza = self.agente_limpieza.ejecutar(df_inicial)
            
            # FASE 3 SECUENCIAL: Los datos limpios van al Agente 3 para entrenar y predecir 
            resultados_modelos = self.agente_prediccion.ejecutar(df_limpio, metadatos_limpieza)
            
            # Agrupamos absolutamente todas las respuestas en un solo paquete estructurado
            salida_sistema = {
                "estado": "Éxito",
                "eda_stats": estadisticas,
                "eda_nulos": nulos,
                "grafico_histograma": path_hist,
                "grafico_correlacion": path_corr,
                "limpieza_meta": metadatos_limpieza,
                "modelos": resultados_modelos
            }
            print(f"================ [{self.nombre}]: PIPELINE FINALIZADO CON ÉXITO ================\n")
            return salida_sistema
            
        except Exception as e:
            # MANEJO DE ERRORES: Si algo llega a fallar adentro de cualquier agente,
            # el Orquestador atrapa el error y evita que la página web se caiga (pantallazo azul).
            print(f"❌ [{self.nombre}]: Hubo un problema en el flujo: {str(e)}")
            return {
                "estado": "Error",
                "mensaje": str(e)
            }