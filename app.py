import streamlit as st
import pandas as pd
# Importamos las funciones mapeadas a nuestros agentes individuales
from agentes import agente_eda, agente_limpieza, agente_prediccion

# Configuración inicial de la página
st.set_page_config(page_title="Boutique - Sistema Multiagente", layout="wide")

# ==========================================
# 🎨 DISEÑO PERSONALIZADO: ROSAS Y MORADOS PASTEL (CSS)
# ==========================================
st.markdown("""
    <style>
    /* Fondo general de la aplicación */
    .stApp {
        background-color: #FAF6FB;
    }
    
    /* Títulos principales */
    h1 {
        color: #6C4A73 !important; /* Morado elegante */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #8E6C99 !important; /* Lila pastel oscuro */
    }
    
    /* Estilo de las pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #F1E5F4; /* Fondo lila muy claro para la barra */
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #FFFFFF; 
        border-radius: 8px;
        color: #8E6C99; /* Texto lila */
        font-weight: bold;
        border: 1px solid #E2CCE6;
        transition: all 0.3s ease;
    }
    /* Pestaña seleccionada activa */
    .stTabs [aria-selected="true"] {
        background-color: #FFD3EC !important; /* Rosa pastel encendido */
        color: #58335E !important;
        border: 1px solid #FFB7E2 !important;
        transform: scale(1.02);
    }
    
    /* Cajas de Información e Indicadores */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #F7D6ED;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(223, 194, 227, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #8E6C99 !important;
        font-weight: 600;
    }
    
    /* Mensajes de Alertas (Success / Info) */
    .stAlert {
        background-color: #FFF0F9 !important; /* Fondo rosa pálido */
        color: #853E6B !important;
        border: 1px solid #FFD3EC !important;
        border-radius: 12px;
    }
    
    /* Líneas divisorias */
    hr {
        border-top: 2px dashed #E2CCE6;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🛍️ INTERFAZ DE USUARIO
# ==========================================
st.title("🛍️ Akasia Boutique: Sistema Multiagente de Ventas")
st.markdown("### ✨ Unidad IV - Inteligencia Artificial | Proyecto Final")
st.write("Bienvenida al software de gestión inteligente. Carga el dataset de la boutique para iniciar el pipeline automatizado.")

# Orquestador: Componente para cargar archivos del usuario
uploaded_file = st.file_uploader("Selecciona el archivo de inventario/ventas (CSV o XLSX)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # El orquestador lee y estructura los datos iniciales
    try:
        if uploaded_file.name.endswith('.csv'):
            df_inicial = pd.read_csv(uploaded_file)
        else:
            df_inicial = pd.read_excel(uploaded_file)
            
        # ✨ NORMALIZACIÓN AUTOMÁTICA DEL DATASET DE SOFI ✨
        # Si encuentra 'Total_Venta', lo renombra a 'Ventas' para acoplarlo al pipeline sin romper nada
        if 'Total_Venta' in df_inicial.columns and 'Ventas' not in df_inicial.columns:
            df_inicial = df_inicial.rename(columns={'Total_Venta': 'Ventas'})
            
        st.success("✨ ¡Archivo recibido y normalizado con éxito por el Agente Orquestador!")
        
        # --- ORQUESTACIÓN Y FLUJO SECUENCIAL ---
        # 1. El orquestador invoca al Agente 1 (EDA) enviando el dataset original
        estadisticas, nulos, f_hist, f_corr = agente_eda(df_inicial)
        
        # 2. El orquestador invoca al Agente 2 (Limpieza) para preparar los datos
        df_limpio_codificado = agente_limpieza(df_inicial)
        
        # 3. El orquestador pasa el dataset limpio al Agente 3 (Predicción)
        mejor_mod, r_lr, r_dt, r_ganador, tabla_pred = agente_prediccion(df_limpio_codificado)
        
        # --- DISEÑO DE LA INTERFAZ DE USUARIO (Pestañas de visualización) ---
        tab1, tab2, tab3 = st.tabs(["📊 Agente 1: EDA", "🧹 Agente 2: Limpieza", "🔮 Agente 3: Predicción"])
        
        with tab1:
            st.header("🔍 Análisis Exploratorio de Datos (EDA)")
            st.write("Métricas descriptivas de la tienda calculadas de forma automatizada:")
            
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
            st.info("🔮 **Estado del Agente 2:** ¡Procesamiento completado! Se eliminaron registros duplicados, se imputaron valores faltantes (como la Edad y el Precio con la mediana) y se ignoraron IDs/Fechas pesadas para el modelo.")
            st.write("**Muestra del Dataset limpio y codificado (Primeras filas listas para entrenar):**")
            st.dataframe(df_limpio_codificado.head(10), use_container_width=True)
            
        with tab3:
            st.header("📈 Evaluación del Modelo Predictivo")
            st.write("El Agente 3 entrenó los modelos matemáticos y seleccionó el de menor $RMSE$ (Error Cuadrático Medio):")
            
            # Métricas destacadas en pantalla
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(label="🏆 Modelo Recomendado", value=mejor_mod)
            col_m2.metric(label="RMSE - Regresión Lineal", value=f"{r_lr:.4f}")
            col_m3.metric(label="RMSE - Árbol de Decisión", value=f"{r_dt:.4f}")
            
            st.write("---")
            st.write("👗 **Predicciones Estimadas de Ventas (Comparativa Real vs Predicción):**")
            st.dataframe(tabla_pred.head(15), use_container_width=True)
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo en el pipeline: {e}")