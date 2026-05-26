import streamlit as st
import pandas as pd
# ✨ CORRECCIÓN: Ahora importamos la clase Maestra del Orquestador ✨
from agentes import AgenteOrquestador

# Configuración inicial de la página
st.set_page_config(page_title="Boutique - Sistema Multiagente", layout="wide")

# Instanciamos al jefe de jefes en la memoria de la interfaz
if 'orquestador' not in st.session_state:
    st.session_state.orquestador = AgenteOrquestador()

# ==========================================
# 🎨 DISEÑO ULTRA PERSONALIZADO PASTEL (SIN NEGROS)
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
st.write("Bienvenida al software de gestión inteligente. Carga el dataset de la boutique para iniciar el pipeline automatizado.")

# Caja de Carga de Archivos (Ubicada en la parte superior)
uploaded_file = st.file_uploader("Selecciona el archivo de ventas (.csv o .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Creamos un archivo temporal local con lo que subió el usuario para que el orquestador pueda leer su ruta
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # ✨ EL ORQUESTADOR ENTRA EN ACCIÓN Y PONE A TRABAJAR A TODOS EN CADENA ✨
        salida = st.session_state.orquestador.procesar_pipeline(temp_path)
        
        if salida["estado"] == "Éxito":
            st.success("✨ ¡Pipeline ejecutado con éxito a través de la arquitectura de clases del Orquestador!")
            
            # --- DISEÑO DE LAS PESTAÑAS EN COLORES PASTEL ---
            tab1, tab2, tab3 = st.tabs(["📊 Agente 1: EDA", "🧹 Agente 2: Limpieza", "🔮 Agente 3: Predicción"])
            
            with tab1:
                st.header("🔍 Análisis Exploratorio de Datos (EDA)")
                st.write("Métricas descriptivas generales calculadas por el `AgenteEDA`:")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(salida["eda_stats"], use_container_width=True)
                with col2:
                    st.write("**Resumen de Registros Nulos Encontrados:**")
                    st.dataframe(salida["eda_nulos"], use_container_width=True)
                    
                st.write("---")
                st.write("💖 **Visualizaciones Gráficas generadas (Cargadas desde almacenamiento estático):**")
                col3, col4 = st.columns(2)
                with col3:
                    st.image(salida["grafico_histograma"], caption="Distribución de Frecuencia de Ventas")
                with col4:
                    st.image(salida["grafico_correlacion"], caption="Matriz de Intercorrelación de Características")
                    
            with tab2:
                st.header("🧼 Limpieza y Preparación de Datos")
                st.info(f"🔮 **Estado del AgenteLimpieza:** Se eliminaron un total de **{salida['limpieza_meta']['duplicados_eliminados']}** filas duplicadas. Se aislaron las variables irrelevantes de alta cardinalidad (Fechas/IDs) y se dummificó la columna de categorías estables.")
                st.write("**Reporte de Categorías Mapeadas Exitosamente:**")
                st.json(salida["limpieza_meta"]["mapeo_categorias"])
                
            with tab3:
                st.header("📈 Inteligencia Predictiva y Demanda")
                st.write("El `AgentePrediccion` ejecutó un modelado comparativo evaluando mediante la métrica $RMSE$:")
                
                # Tarjetas de Métricas en color ciruela elegante
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric(label="🏆 Modelo de Máximo Rendimiento", value=salida["modelos"]["modelo_seleccionado"])
                col_m2.metric(label="RMSE - Regresión Lineal", value=f"{salida['modelos']['rmse_regresion_lineal']:.4f}")
                col_m3.metric(label="RMSE - Árbol de Decisión", value=f"{salida['modelos']['rmse_arbol_decision']:.4f}")
                
                st.write("---")
                st.write("👗 **Ventas Promedio Estimadas en Pesos ($) por Categoría Comercial:**")
                # Convertimos el diccionario de predicciones a una tabla bonita
                df_pred_cat = pd.DataFrame(list(salida["modelos"]["predicciones_categoria"].items()), columns=["Categoría de Ropa", "Venta Estimada ($)"])
                st.dataframe(df_pred_cat, use_container_width=True)
                
        else:
            st.error(f"El Orquestador detuvo el pipeline por seguridad: {salida['mensaje']}")
            
    except Exception as e:
        st.error(f"Error crítico en el renderizado de la UI: {e}")