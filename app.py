import streamlit as st
import pandas as pd
# Importamos las funciones mapeadas a nuestros agentes individuales
from agentes import agente_eda, agente_limpieza, agente_prediccion

# Configuración inicial de la página
st.set_page_config(page_title="Boutique - Sistema Multiagente", layout="wide")

# ==========================================
# 🎨 DISEÑO PERSONALIZADO: MÁXIMA ARMONÍA PASTEL (SIN NEGROS)
# ==========================================
st.markdown("""
    <style>
    .stApp {
        background-color: #FEFBE7; 
    }
    .stApp, p, span, label {
        color: #4A2552 !important; 
        font-weight: 500;
    }
    h1 {
        color: #6C4A73 !important; 
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    h2, h3, .stMarkdown h2, .stMarkdown h3 {
        color: #8E6C99 !important; 
        font-weight: 700 !important;
    }
    header[data-testid="stHeader"] {
        background-color: #FEFBE7 !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important; 
        border: 2px dashed #E2CCE6 !important; 
        border-radius: 15px !important;
    }
    div[data-testid="stFileUploaderDropzone"] p {
        color: #8E6C99 !important; 
    }
    button[data-testid="stBaseButton-secondary"] {
        background-color: #FFF0F9 !important; 
        color: #6C4A73 !important;
        border: 1px solid #FFD3EC !important;
    }
    div[data-testid="stTable"] th, .styled-table th {
        background-color: #F1E5F4 !important;
        color: #4A2552 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #F1E5F4; 
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #FFFFFF; 
        border-radius: 8px;
        color: #8E6C99 !important; 
        font-weight: 800 !important;
        border: 1px solid #E2CCE6;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFD3EC !important; 
        color: #58335E !important; 
        border: 2px solid #FFB7E2 !important;
        transform: scale(1.02);
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF; 
        border: 2px solid #F7D6ED;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(223, 194, 227, 0.2);
    }
    div[data-testid="stMetricValue"] {
        color: #6C4A73 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8E6C99 !important;
        font-weight: 700 !important;
    }
    .stAlert {
        background-color: #FFF0F9 !important; 
        color: #6C4A73 !important; 
        border: 2px solid #FFD3EC !important;
        border-radius: 12px;
    }
    hr {
        border-top: 2px dashed #E2CCE6;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🛍️ INTERFAZ DE USUARIO
# ==========================================
st.title("🛍️ Boutique: Sistema Multiagente de Ventas")
st.markdown("### ✨ Unidad IV - Inteligencia Artificial")
st.write("Carga cualquier dataset de ventas en formato .csv o .xlsx. El Orquestador identificará automáticamente las variables críticas.")

uploaded_file = st.file_uploader("Selecciona el archivo de ventas (.csv o .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_inicial = pd.read_csv(uploaded_file)
        else:
            df_inicial = pd.read_excel(uploaded_file)
            
        # ✨ DETECTOR INTELIGENTE Y UNIVERSAL DE COLUMNAS DE VENTAS ✨
        posibles_nombres_ventas = ['ventas', 'venta', 'total_venta', 'total_ventas', 'sales', 'total_sales', 'monto']
        columna_ventas_encontrada = None
        
        for col in df_inicial.columns:
            if col.lower().strip() in posibles_nombres_ventas:
                columna_ventas_encontrada = col
                break
        
        if columna_ventas_encontrada:
            df_inicial = df_inicial.rename(columns={columna_ventas_encontrada: 'Ventas'})
            st.success(f"✨ ¡Archivo procesado! Se detectó la columna objetivo bajo el nombre de: '{columna_ventas_encontrada}'")
        else:
            # Si el archivo tiene un nombre rarísimo que no mapeamos, le pedimos al usuario que nos diga cuál es
            lista_columnas = df_inicial.columns.tolist()
            col_seleccionada = st.selectbox("No logré identificar la columna de ventas automáticamente. Por favor, selecciónala tú misma:", lista_columnas)
            df_inicial = df_inicial.rename(columns={col_seleccionada: 'Ventas'})
            st.success("✨ ¡Columna objetivo asignada manualmente por el usuario!")

        # --- ORQUESTACIÓN Y FLUJO SECUENCIAL ---
        estadisticas, nulos, f_hist, f_corr = agente_eda(df_inicial)
        df_limpio_codificado = agente_limpieza(df_inicial)
        mejor_mod, r_lr, r_dt, r_ganador, tabla_pred = agente_prediccion(df_limpio_codificado)
        
        # --- DISEÑO DE LAS PESTAÑAS PASTEL ---
        tab1, tab2, tab3 = st.tabs(["📊 Agente 1: EDA", "🧹 Agente 2: Limpieza", "🔮 Agente 3: Predicción"])
        
        with tab1:
            st.header("🔍 Análisis Exploratorio de Datos (EDA)")
            st.write("Métricas descriptivas del archivo calculadas automáticamente:")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(estadisticas, use_container_width=True)
            with col2:
                st.write("**Conteo de valores nulos:**")
                st.dataframe(nulos, use_container_width=True)
            st.write("---")
            st.write("💖 **Visualizaciones gráficas generadas por el Agente 1:**")
            col3, col4 = st.columns(2)
            with col3:
                st.pyplot(f_hist)
            with col4:
                st.pyplot(f_corr)
                
        with tab2:
            st.header("🧼 Limpieza y Codificación de Datos")
            st.info("🔮 **Estado del Agente 2:** ¡Procesamiento completado! Se eliminaron registros duplicados, se imputaron nulos y se dummificaron categorías estables.")
            st.write("**Muestra del Dataset limpio y codificado (Primeras filas listas para entrenar):**")
            st.dataframe(df_limpio_codificado.head(10), use_container_width=True)
            
        with tab3:
            st.header("📈 Evaluación del Modelo Predictivo")
            st.write("El Agente 3 entrenó los modelos matemáticos y seleccionó el de menor $RMSE$ (Error Cuadrático Medio):")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(label="🏆 Modelo Recomendado", value=mejor_mod)
            col_m2.metric(label="RMSE - Regresión Lineal", value=f"{r_lr:.4f}")
            col_m3.metric(label="RMSE - Árbol de Decisión", value=f"{r_dt:.4f}")
            
            st.write("---")
            st.write("👗 **Predicciones Estimadas de Ventas (Comparativa Real vs Predicción):**")
            st.dataframe(tabla_pred.head(15), use_container_width=True)
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo en el pipeline: {e}")