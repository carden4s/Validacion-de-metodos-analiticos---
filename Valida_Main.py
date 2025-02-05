import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, f_oneway
import streamlit as st
from io import BytesIO
from pathlib import Path
from matplotlib.figure import Figure
import plotly.express as px
from fpdf import FPDF
import base64
from datetime import datetime
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Obtener la ruta del directorio actual
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
imagenes_dir = current_dir / "img"

# Page Configuration
st.set_page_config(
    page_title="Validación UV-Vis",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Escudo_CUCEI.svg/424px-Escudo_CUCEI.svg.png",
    layout="wide"  # Centers all content
)
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS for title and footer
st.markdown("""
    <style>
    /* Title container styling */
    .title-container {
        display: flex;
        justify-content: center; /* Center the title and logo */
        align-items: center;
        gap: 10px;
        margin-bottom: 20px; /* Spacing below title */
    }
    .title-container img:first-child {
        width: 120px; /* Adjust first icon size */
        height: auto;
    }
    .title-container img:last-child {
        width: 200px; /* Adjust second icon size */
        height: auto;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.05); /* Light background */
        color: gray;
        text-align: center;
        padding: 5px 15px;
        font-size: small;
        opacity: 0.8;
    }
    </style>
""", unsafe_allow_html=True)

# Title with Icon
st.markdown("""
    <div class="title-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Escudo_UdeG.svg/662px-Escudo_UdeG.svg.png" alt="UDG Logo">
        <h1>Validación de Métodos Analíticos - Espectrofotometría UV-Vis</h1>
        <img src="https://practicas.cucei.udg.mx/dist/imagenes/logo_cucei_blanco.png" alt="CUCEI Logo">
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        2025 - Luis Angel Cardenas Medina
    </div>
""", unsafe_allow_html=True)


# Módulos disponibles
modulo = st.sidebar.selectbox(
    "Selecciona el módulo de análisis:",
    [
        "Linealidad y Rango",
        "Límites de Detección y Cuantificación",
        "Exactitud (Recuperación)",
        "Precisión (Repetibilidad e Intermedia)",
        "Robustez",
        "Estabilidad"
    ]
)

# Funciones generales

def previsualizar_datos(datos):
    """Previsualiza los datos cargados en la interfaz."""
    st.write("### Vista previa de los datos cargados:")
    st.write(datos.head())
    st.write(f"**Número de filas:** {datos.shape[0]}, **Número de columnas:** {datos.shape[1]}")

def validar_columnas(datos, columnas_necesarias):
    """Valida que las columnas requeridas estén presentes en el archivo subido."""
    columnas_faltantes = [col for col in columnas_necesarias if col not in datos.columns]
    if columnas_faltantes:
        st.error(f"El archivo no contiene las siguientes columnas necesarias: {', '.join(columnas_faltantes)}")
        return False
    return True




def calcular_linealidad(datos):
    """Calcula la linealidad y rango del método considerando agrupación por días."""
    # Configuración inicial de estilo
    sns.set_theme(style="whitegrid", palette="muted")
    COLORS = ['#2ecc71', '#e74c3c']  # Verde y rojo para temas
    
    # Validación mejorada
    columnas_necesarias = ['Concentración', 'Absorbancia', 'Tipo', 'Día']
    if not validar_columnas(datos, columnas_necesarias):
        return
    
    # Chequear valores numéricos
    if not np.issubdtype(datos['Concentración'].dtype, np.number) or \
       not np.issubdtype(datos['Absorbancia'].dtype, np.number):
        st.error("Las columnas 'Concentración' y 'Absorbancia' deben ser numéricas")
        return

    for dia, grupo_dia in datos.groupby('Día'):
        with st.container():
            st.markdown(f"## 📅 Día {dia}")
            
            # Sección de estándares
            estandar = grupo_dia[grupo_dia['Tipo'] == 'Estándar']
            if estandar.empty:
                st.warning(f"⚠️ No se encontraron datos de Estándar para el Día {dia}.")
                continue
                
            # Procesamiento de datos
            estandar_promedio = estandar.groupby('Concentración')['Absorbancia'].mean().reset_index()
            
            try:
                # Cálculo de regresión con manejo de errores
                regresion = linregress(estandar_promedio['Concentración'], estandar_promedio['Absorbancia'])
                slope = regresion.slope
                intercept = regresion.intercept
                r_value = regresion.rvalue
                p_value = regresion.pvalue
                predicciones = slope * estandar_promedio['Concentración'] + intercept
                residuales = estandar_promedio['Absorbancia'] - predicciones
            except Exception as e:
                st.error(f"❌ Error en análisis de regresión: {str(e)}")
                continue

            # Mostrar métricas en columnas
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Métricas de Regresión")
                st.metric("Coeficiente de Determinación (R²)", f"{r_value**2:.4f}")
                st.metric("Pendiente (Slope)", f"{slope:.4f}")
                st.metric("Intercepto", f"{intercept:.4f}")
                
            with col2:
                st.markdown("### 📈 Evaluación de Calidad")
                st.metric("Coeficiente de Correlación (R)", f"{r_value:.4f}")
                st.metric("Valor p", f"{p_value:.4e}")
                cumplimiento = r_value**2 >= 0.995
                st.metric("Cumplimiento ICH Q2(R² ≥ 0.995)", 
                        "✅ Cumple" if cumplimiento else "❌ No Cumple",
                        delta=f"{r_value**2 - 0.995:.4f}" if not cumplimiento else None)

            # Gráficos profesionales con estilo unificado
            fig = plt.figure(figsize=(14, 6), facecolor='#f8f9fa')
            gs = fig.add_gridspec(1, 2)
            
            # Gráfico de Regresión
            ax1 = fig.add_subplot(gs[0, 0])
            sns.regplot(x=estandar_promedio['Concentración'], y=estandar_promedio['Absorbancia'], 
                        ax=ax1, ci=95, scatter_kws={'s': 80, 'edgecolor': 'black', 'alpha': 0.8},
                        line_kws={'color': COLORS[0], 'lw': 2, 'alpha': 0.8})
            ax1.set_title(f"Regresión Lineal - Día {dia}", fontsize=14, pad=20)
            ax1.set_xlabel("Concentración (μg/mL)", fontsize=12)
            ax1.set_ylabel("Absorbancia (UA)", fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Anotaciones en el gráfico
            textstr = '\n'.join((
                f'$R^2 = {r_value**2:.4f}$',
                f'$y = {slope:.4f}x + {intercept:.4f}$'))
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Gráfico de Residuales
            ax2 = fig.add_subplot(gs[0, 1])
            residual_plot = sns.residplot(x=estandar_promedio['Concentración'], y=residuales,
                                        ax=ax2, lowess=True, 
                                        scatter_kws={'s': 60, 'color': COLORS[1], 'edgecolor': 'black'},
                                        line_kws={'color': COLORS[0], 'lw': 2})
            ax2.axhline(0, color='black', linestyle='--', lw=1.5)
            ax2.set_title(f"Análisis de Residuales - Día {dia}", fontsize=14, pad=20)
            ax2.set_xlabel("Concentración (μg/mL)", fontsize=12)
            ax2.set_ylabel("Residuales", fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Destacar outliers
            outliers = np.abs(residuales) > 2 * residuales.std()
            if outliers.any():
                ax2.scatter(estandar_promedio['Concentración'][outliers], 
                          residuales[outliers], 
                          s=100, edgecolor='black', 
                          facecolor='none', linewidth=1.5,
                          label='Outliers (±2σ)')
                ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Análisis de muestras con mejor visualización
            muestra = grupo_dia[grupo_dia['Tipo'] == 'Muestra']
            if not muestra.empty:
                with st.expander(f"🧪 Resultados de Muestras - Día {dia}", expanded=False):
                    try:
                        muestra['Concentración Estimada'] = (muestra['Absorbancia'] - intercept) / slope
                        
                        # Gráfico de resultados
                        fig_m = plt.figure(figsize=(10, 5))
                        sns.scatterplot(x=muestra['Concentración Estimada'], 
                                      y=muestra['Absorbancia'],
                                      s=100, edgecolor='black',
                                      color=COLORS[0], alpha=0.8)
                        plt.title(f"Concentraciones Estimadas - Día {dia}", fontsize=14)
                        plt.xlabel("Concentración Estimada (μg/mL)", fontsize=12)
                        plt.ylabel("Absorbancia (UA)", fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig_m)
                        plt.close(fig_m)
                        
                        # Tabla de resultados
                        st.dataframe(
                            muestra[['Absorbancia', 'Concentración Estimada']]
                            .style.format("{:.4f}")
                            .highlight_between(subset=['Concentración Estimada'], 
                                             color='#f8d7da',  # Rojo claro
                                             props='color: #721c24;',  # Texto oscuro
                                             axis=None)
                        )
                    except ZeroDivisionError:
                        st.error("Error: Pendiente cero, no se puede calcular concentración")

            # Sección de interpretación interactiva
            with st.expander("🔍 Guía de Interpretación", expanded=False):
                st.markdown("""
                **Análisis de Residuales:**
                - ✅ **Distribución aleatoria:** Buen ajuste del modelo
                - ⚠️ **Patrón no lineal:** Considerar modelo no lineal
                - ❌ **Efecto embudo:** Varianza no constante
                - 📌 **Outliers:** Verificar mediciones sospechosas

                **Criterios ICH Q2:**
                - $R^2 ≥ 0.995$ para validación
                - Residuales < ±2σ (95% confianza)
                """)

def calcular_lod_loq(datos):
    """Calcula LOD y LOQ con visualización mejorada y validación extendida."""
    # Configuración de estilo
    COLORS = ['#3498db', '#2ecc71', '#e74c3c']  # Azul, Verde, Rojo
    sns.set_theme(style="whitegrid", font_scale=0.95)
    
    # Validación mejorada
    columnas_necesarias = ['Día', 'Tipo', 'Concentración', 'Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return
    
    # Chequear tipos numéricos
    if not np.issubdtype(datos['Concentración'].dtype, np.number) or \
       not np.issubdtype(datos['Absorbancia'].dtype, np.number):
        st.error("❌ Las columnas 'Concentración' y 'Absorbancia' deben ser numéricas")
        return

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    if datos_estandar.empty:
        st.error("❌ No se encontraron datos de tipo 'Estándar'")
        return

    with st.expander("📊 Método de Cálculo ICH Q2", expanded=True):
        st.markdown("""
        **Fórmulas aplicadas:**
        - $LOD = \\frac{3.3 \\times σ}{S}$  
        - $LOQ = \\frac{10 \\times σ}{S}$  
        Donde:
        - σ: Desviación estándar de los residuales
        - S: Pendiente de la curva de calibración
        """)

    dias_unicos = datos_estandar['Día'].unique()
    for dia in dias_unicos:
        with st.container():
            st.markdown(f"## 📅 Día {dia}")
            datos_dia = datos_estandar[datos_estandar['Día'] == dia]
            
            if len(datos_dia) < 3:
                st.warning(f"⚠️ Mínimo 3 puntos requeridos para cálculo confiable (Día {dia})")
                continue
                
            try:
                # Cálculo completo de regresión
                regresion = linregress(datos_dia['Concentración'], datos_dia['Absorbancia'])
                slope = regresion.slope
                intercept = regresion.intercept
                r_value = regresion.rvalue
                r_squared = r_value**2
                residuals = datos_dia['Absorbancia'] - (slope * datos_dia['Concentración'] + intercept)
                std_dev = residuals.std()
                lod = (3.3 * std_dev) / slope if slope != 0 else None
                loq = (10 * std_dev) / slope if slope != 0 else None
                
                # Mostrar métricas en columnas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pendiente (S)", f"{slope:.4f}")
                    st.metric("R²", f"{r_squared:.4f}")
                    
                with col2:
                    st.metric("LOD", f"{lod:.4f}", help="3.3σ/S")
                    st.metric("LOQ", f"{loq:.4f}", help="10σ/S")
                
                with col3:
                    rango_dinamico = datos_dia['Concentración'].max() / lod if lod else 0
                    st.metric("Rango Dinámico", 
                            f"{rango_dinamico:.1f}:1", 
                            help="Relación LOQ:LOD recomendada ≥ 3:1")
                    st.metric("Cumplimiento ICH", 
                            "✅" if rango_dinamico >= 3 else "❌", 
                            delta="≥3:1" if rango_dinamico >=3 else None)

                # Gráfico de datos brutos
                with st.expander(f"🔍 Datos Detallados - Día {dia}", expanded=False):
                    fig = plt.figure(figsize=(10, 4))
                    ax = fig.add_subplot(111)
                    sns.scatterplot(data=datos_dia, x='Concentración', y='Absorbancia',
                                   s=100, color=COLORS[0], edgecolor='black',
                                   ax=ax, label='Datos Experimentales')
                    ax.set_title(f"Datos Crudos - Día {dia} (R² = {r_squared:.4f})", fontsize=14)
                    ax.set_xlabel("Concentración (μg/mL)", fontsize=12)
                    ax.set_ylabel("Absorbancia (UA)", fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    plt.close(fig)
                    
            except Exception as e:
                st.error(f"❌ Error en Día {dia}: {str(e)}")
                continue

def graficar_curva_calibracion_streamlit(datos):
    """Grafica curva de calibración con estilo profesional y anotaciones."""
    # Configuración de estilo
    COLORS = ['#2ecc71', '#3498db', '#e74c3c']  # Verde, Azul, Rojo
    plt.style.use('seaborn-talk')
    
    # Validación de datos
    columnas_necesarias = ['Día', 'Tipo', 'Concentración', 'Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    if datos_estandar.empty:
        st.error("❌ No se encontraron datos de calibración")
        return

    dias_unicos = datos_estandar['Día'].unique()
    for dia in dias_unicos:
        with st.container():
            st.markdown(f"## 📈 Curva de Calibración - Día {dia}")
            datos_dia = datos_estandar[datos_estandar['Día'] == dia]
            
            if len(datos_dia) < 3:
                st.warning(f"⚠️ Mínimo 3 puntos recomendados para curva confiable (Día {dia})")
                continue
                
            try:
                # Cálculo completo de regresión
                regresion = linregress(datos_dia['Concentración'], datos_dia['Absorbancia'])
                slope = regresion.slope
                intercept = regresion.intercept
                r_value = regresion.rvalue
                r_squared = r_value**2
                residuals = datos_dia['Absorbancia'] - (slope * datos_dia['Concentración'] + intercept)
                std_dev = residuals.std()
                lod = (3.3 * std_dev) / slope if slope != 0 else None
                loq = (10 * std_dev) / slope if slope != 0 else None
                
                # Crear figura profesional
                fig = plt.figure(figsize=(10, 6), facecolor='#f8f9fa')
                ax = fig.add_subplot(111)
                
                # Gráfico principal
                sns.regplot(x=datos_dia['Concentración'], y=datos_dia['Absorbancia'],
                           ax=ax, ci=95,
                           scatter_kws={'s': 80, 'color': COLORS[0], 'edgecolor': 'black', 'alpha': 0.8},
                           line_kws={'color': COLORS[1], 'lw': 2, 'alpha': 0.8})
                
                # Líneas de LOD/LOQ
                ax.axvline(lod, color=COLORS[2], linestyle='--', lw=2, alpha=0.8, label=f'LOD ({lod:.2f})')
                ax.axvline(loq, color=COLORS[1], linestyle='-.', lw=2, alpha=0.8, label=f'LOQ ({loq:.2f})')
                
                # Anotaciones profesionales
                textstr = '\n'.join((
                    f'$R^2 = {r_squared:.4f}$',
                    f'$y = {slope:.4f}x + {intercept:.4f}$',
                    f'σ = {std_dev:.4f}'))
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Estilo del gráfico
                ax.set_title(f"Curva de Calibración - Día {dia}", fontsize=16, pad=20)
                ax.set_xlabel("Concentración (μg/mL)", fontsize=14)
                ax.set_ylabel("Absorbancia (UA)", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='lower right', frameon=True, shadow=True)
                
                # Resaltar área LOD/LOQ
                ax.axvspan(0, loq, facecolor='#f8d7da', alpha=0.3, label='Zona LOD/LOQ')
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Tabla de datos adjunta
                with st.expander(f"📋 Datos de Calibración - Día {dia}"):
                    st.dataframe(
                        datos_dia[['Concentración', 'Absorbancia']]
                        .style.format("{:.4f}")
                        .highlight_between(subset=['Concentración'], 
                                         left=0, right=loq,
                                         color='#fff3cd')  # Amarillo claro
                    )
                    
            except Exception as e:
                st.error(f"❌ Error en Día {dia}: {str(e)}")
                continue
def generar_reporte_ich(estandares, muestras):
    """Genera reporte PDF/Excel con resultados detallados"""
    # Implementación personalizada según necesidades
    pass

def calcular_precision(datos):
    """
    Evalúa la precisión según ICH Q2 R1 para UV -Vis.
    """
    # Configuración inicial
    sns.set_theme(style="whitegrid", palette="muted")
    st.header("🎯 Análisis de Precisión - UV-Vis")

    # Validación y limpieza de datos
    for col in ['Concentración', 'Absorbancia']:
        datos[col] = datos[col].replace('.', np.nan)
        datos[col] = pd.to_numeric(datos[col], errors='coerce')
    datos.dropna(subset=['Concentración', 'Absorbancia'], inplace=True)

    # Separar datos
    estandares = datos[datos['Tipo'] == 'Estándar'].copy()
    muestras = datos[datos['Tipo'] == 'Muestra'].copy()

    # Validación de réplicas mínimas
    conteo_replicas = estandares.groupby(['Día', 'Concentración']).size()
    if any(conteo_replicas < 3):
        st.error("❌ ICH Q2: Se requieren mínimo 3 réplicas por concentración y día")
        return

    # Procesamiento de curvas de calibración
    dias_validos = []
    resultados_muestras = []
    with st.spinner('Procesando curvas de calibración...'):
        for dia in estandares['Día'].unique():
            try:
                est_dia = estandares[estandares['Día'] == dia]
                slope, intercept, r_value, _, _ = linregress(
                    est_dia['Absorbancia'], est_dia['Concentración']
                )
                if r_value**2 < 0.995:
                    st.warning(f"🚨 Día {dia} excluido - R²: {r_value**2:.3f}")
                    continue
                mues_dia = muestras[muestras['Día'] == dia].copy()
                mues_dia['Conc. Calculada'] = slope * mues_dia['Absorbancia'] + intercept
                resultados_muestras.append(mues_dia)
                dias_validos.append(dia)
            except Exception as e:
                st.error(f"Error en el Día {dia}: {str(e)}")

    # Análisis de Estándares
    st.subheader("🔬 Resultados para Estándares")
    col1, col2 = st.columns(2)
    with col1:
        grupos_intra = estandares.groupby(['Día', 'Concentración'])['Absorbancia']
        rsd_intra = (grupos_intra.std() / grupos_intra.mean() * 100).reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=rsd_intra, x='Concentración', y='Absorbancia', palette='Blues', ax=ax)
        ax.axhline(2, color='red', linestyle='--', label='Límite ICH (2%)')
        ax.set_title("RSD Intraensayo por Concentración")
        ax.set_ylabel("RSD (%)")
        ax.legend()
        st.pyplot(fig)
        max_rsd_intra = rsd_intra['Absorbancia'].max()
        st.metric("RSD Máximo Intraensayo", f"{max_rsd_intra:.2f}%",
                 delta="Cumple" if max_rsd_intra <= 2 else "No Cumple")
    with col2:
        grupos_inter = estandares.groupby('Concentración')['Absorbancia']
        rsd_inter = (grupos_inter.std() / grupos_inter.mean() * 100).reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=rsd_inter, x='Concentración', y='Absorbancia',
                    marker='o', color='green', linewidth=2, ax=ax)
        ax.axhline(3, color='red', linestyle='--', label='Límite ICH (3%)')
        ax.set_title("RSD Intermedio por Concentración")
        ax.set_ylabel("RSD (%)")
        ax.legend()
        st.pyplot(fig)
        rsd_ponderado = (rsd_inter['Absorbancia'] * grupos_inter.mean()).sum() / grupos_inter.mean().sum()
        st.metric("RSD Ponderado Intermedio", f"{rsd_ponderado:.2f}%",
                 delta="Cumple" if rsd_ponderado <= 3 else "No Cumple")

    # Resumen Ejecutivo
    with st.expander("📊 Resumen", expanded=True):
        metricas = {
            'Días Analizados': len(dias_validos),
            'RSD Intraensayo Máximo': f"{max_rsd_intra:.2f}%",
            'RSD Intermedio Ponderado': f"{rsd_ponderado:.2f}%",
            'Muestras Válidas': len(resultados_muestras) if resultados_muestras else 0,
            'Cumplimiento Global': "✅" if (max_rsd_intra <= 2 and rsd_ponderado <= 3) else "❌"
        }
        st.json(metricas)
        
def calcular_exactitud(datos):
    """Calcula la exactitud mediante recuperación según ICH Q2 usando concentración teórica vs real."""
    # Validar columnas requeridas
    columnas_necesarias = ['Día', 'Concentración Teórica', 'Concentración Real']
    if not validar_columnas(datos, columnas_necesarias):
        return
    
    # Calcular porcentaje de recuperación
    datos['Recuperación (%)'] = (datos['Concentración Real'] / datos['Concentración Teórica']) * 100
    
    # Análisis por día
    st.header("Análisis de Exactitud - ICH Q2 R1")
    
    # Crear pestañas para organización
    tab1, tab2, tab3 = st.tabs(["Resumen Estadístico", "Distribución de Recuperaciones", "Datos Detallados"])
    
    with tab1:
        # Métricas clave por día
        st.subheader("Estadísticos por Día")
        resumen = datos.groupby('Día').agg(
            Muestras=('Recuperación (%)', 'size'),
            Media=('Recuperación (%)', lambda x: f"{x.mean():.2f}%"),
            DE=('Recuperación (%)', lambda x: f"{x.std():.2f}%"),
            RSD=('Recuperación (%)', lambda x: f"{(x.std()/x.mean())*100:.2f}%"),
            Mín=('Recuperación (%)', lambda x: f"{x.min():.2f}%"),
            Máx=('Recuperación (%)', lambda x: f"{x.max():.2f}%")
        ).reset_index()
        
        # Evaluación de criterios ICH
        resumen['Cumplimiento'] = datos.groupby('Día').apply(
            lambda x: "✅" if (98 <= x['Recuperación (%)'].mean() <= 102) and 
                            (x['Recuperación (%)'].std() <= 3) else "❌"
        ).values
        
        st.dataframe(
            resumen.style.apply(lambda x: ['background: #e6f4ea' if x.Cumplimiento == '✅' 
                                          else 'background: #fce8e6' for _ in x], axis=1)
        )
        
        # Leyenda de cumplimiento
        st.markdown("""
        **Criterios ICH Q2 R1:**
        - Media de recuperación entre 98-102%
        - Desviación estándar ≤3%
        """)
    
    with tab2:
        # Gráfico de distribución
        st.subheader("Distribución de Recuperaciones")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Boxplot por día
        sns.boxplot(data=datos, x='Día', y='Recuperación (%)', palette='viridis', ax=ax)
        ax.axhline(100, color='red', linestyle='--', linewidth=1, label='Objetivo 100%')
        ax.set_title("Distribución de Porcentajes de Recuperación por Día")
        ax.set_ylim(85, 115)
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Histograma combinado
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=datos, x='Recuperación (%)', bins=12, kde=True, 
                    hue='Día', multiple='stack', palette='viridis')
        ax.set_title("Distribución Global de Recuperaciones")
        st.pyplot(fig)
        plt.close(fig)
    
    with tab3:
        # Tabla detallada con formato condicional
        st.subheader("Datos Completos")
        st.dataframe(
            datos.style.format({
                'Concentración Teórica': '{:.4f}',
                'Concentración Real': '{:.4f}',
                'Recuperación (%)': '{:.2f}%'
            }).apply(lambda x: ['color: #2ecc71' if (x['Recuperación (%)'] >= 98) and 
                              (x['Recuperación (%)'] <= 102) else 'color: #e74c3c' 
                              for _ in x], axis=1)
        )
        
        # Botón de descarga
        generar_descarga(datos)

def validar_columnas(datos, columnas):
    """Valida la presencia de columnas requeridas en el dataset"""
    faltantes = [col for col in columnas if col not in datos.columns]
    if faltantes:
        st.error(f"Columnas faltantes: {', '.join(faltantes)}")
        return False
    return True

def generar_descarga(datos):
    """Genera archivo Excel descargable con los resultados"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        datos.to_excel(writer, index=False, sheet_name='Resultados')
    
    st.download_button(
        label="📥 Descargar Resultados Completos",
        data=output.getvalue(),
        file_name="exactitud_analitica.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Descarga todos los resultados en formato Excel"
    )

def evaluar_robustez(datos):
    """Evalúa la robustez del método mediante análisis estadístico avanzado."""
    # Configuración de estilo
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.rcParams['axes.titleweight'] = 'bold'
    st.header("🧪 Análisis de Robustez - ICH Q2")

    # Validación mejorada
    columnas_necesarias = ['Absorbancia', 'Día', 'Concentración', 'Tipo', 'Analista']
    if not validar_columnas(datos, columnas_necesarias):
        return

    with st.expander("📚 Metodología", expanded=False):
        st.markdown("""
        **Criterios ICH Q2 para Robustez:**
        1. ANOVA con nivel de significancia α = 0.05
        2. Tamaño del efecto (η²) < 0.1 para considerar robustez
        3. Mínimo 3 niveles por factor analizado
        4. Intervalos de confianza del 95% para medias
        """)

    # Selección de factores
    factores = st.multiselect(
        "Selecciona factores a evaluar:",
        options=['Día', 'Concentración', 'Tipo', 'Analista'],
        default=['Día']
    )

    # Contenedor principal
    with st.container():
        for factor in factores:
            st.subheader(f"📌 Factor: {factor}")
            col1, col2 = st.columns([1, 2])

            with col1:
                # Análisis estadístico
                grupos = [g['Absorbancia'] for _, g in datos.groupby(factor)]
                
                if len(grupos) < 2:
                    st.error("Se requieren al menos 2 grupos para ANOVA")
                    continue
                
                # ANOVA
                f_stat, p_val = f_oneway(*grupos)
                
                # Tamaño del efecto
                ss_total = np.sum((datos['Absorbancia'] - datos['Absorbancia'].mean())**2)
                eta_squared = f_stat * (len(grupos) - 1) / (len(datos) - len(grupos))
                
                # Resultados
                st.markdown("""
                **Resultados Estadísticos:**
                - Estadístico F: `{:.4f}`
                - Valor p: `{:.4f}`
                - η² (tamaño efecto): `{:.3f}`
                """.format(f_stat, p_val, eta_squared))

                # Evaluación de criterios
                criterio_p = p_val > 0.05
                criterio_eta = eta_squared < 0.1
                robustez = criterio_p and criterio_eta
                
                st.markdown(f"""
                **Interpretación:**
                - Significancia estadística: {"✅ No significativa" if criterio_p else "❌ Significativa"}
                - Tamaño del efecto: {"✅ Aceptable" if criterio_eta else "❌ Excesivo"}
                - **Conclusión:** {"🏆 Método Robusto" if robustez else "⚠️ Requiere atención"}
                """)

            with col2:
                # Visualización avanzada
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=factor, y='Absorbancia', data=datos, 
                           width=0.6, linewidth=1.5, fliersize=4)
                
                # Añadir media e intervalos
                mean_line = datos['Absorbancia'].mean()
                ax.axhline(mean_line, color='red', linestyle='--', 
                          label=f'Media Global: {mean_line:.2f}')
                
                # Formateo profesional
                ax.set_title(f"Distribución por {factor}\n", fontsize=12)
                ax.set_xlabel(factor, fontweight='bold')
                ax.set_ylabel("Absorbancia", fontweight='bold')
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)

                # Análisis post-hoc si es necesario
                if not robustez:
                
                    tukey = pairwise_tukeyhsd(datos['Absorbancia'], datos[factor])
                    with st.expander("🔍 Análisis Post-Hoc (Tukey)"):
                        st.text(str(tukey.summary()))

    # Sección adicional
    with st.expander("📊 Reporte Completo", expanded=False):
        st.subheader("📈 Tendencia Temporal")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=datos, x='Día', y='Absorbancia', 
                    hue='Concentración', style='Tipo', 
                    markers=True, ci=95)
        ax.set_title("Tendencia de Absorbancia por Día y Concentración")
        st.pyplot(fig)
        plt.close(fig)  

def evaluar_estabilidad(datos):
    """
    Evalúa la estabilidad de soluciones por día según ICH Q2 R1 para UV-Vis.
    Se consideran la concentración teórica y el analista como factores adicionales.
    
    Aspectos evaluados:
      - Preprocesamiento: conversión de fechas y columnas numéricas.
      - Análisis de Estándares: 
          * ANOVA entre días y analistas.
          * Cálculo de RSD por rango de concentración (Bajo, Medio, Alto).
          * Gráficos de tendencia y boxplots de Absorbancia.
          * Evaluación de la pendiente de la tendencia (regresión lineal).
          * **Análisis de residuales:** Scatter y violin plots para detectar patrones no aleatorios.
      - Análisis de Muestras: Similar agrupación por rango teórico, gráficos de tendencia y métricas.
    """
    # Configuración inicial
    st.header("📊 Análisis de Estabilidad / Robustez UV-Vis")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Validación de columnas requeridas
    required_cols = ['Día', 'Concentración', 'Absorbancia', 'Tipo', 'Analista']
    if not all(col in datos.columns for col in required_cols):
        st.error(f"❌ Faltan columnas requeridas: {', '.join(required_cols)}")
        return

    try:
        # ============================================
        # 1. Preprocesamiento de Datos
        # ============================================
        with st.expander("🧹 Preprocesamiento de datos", expanded=False):
            df = datos.copy()
            # Convertir 'Día' a datetime (formato dd/mm/YYYY)
            df['Día'] = pd.to_datetime(df['Día'], format='%d/%m/%Y', errors='coerce')
            # Verificar fechas inválidas
            invalid_dates = df[df['Día'].isna()]
            if not invalid_dates.empty:
                st.error("⚠️ Fechas inválidas detectadas:")
                st.dataframe(invalid_dates)
                return
            # Calcular 'Día_num' (días desde la primera medición)
            df['Día_num'] = (df['Día'] - df['Día'].min()).dt.days + 1

            # Convertir columnas numéricas (reemplazando comas por puntos)
            numeric_cols = ['Concentración', 'Absorbancia']
            for col in numeric_cols:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            # Verificar que no haya valores no numéricos
            non_numeric = df[df[numeric_cols].isna().any(axis=1)]
            if not non_numeric.empty:
                st.error("❌ Valores no numéricos detectados:")
                st.dataframe(non_numeric)
                return
            
            st.success("✅ Datos preprocesados correctamente")
            st.dataframe(df.head(), hide_index=True)

        # ============================================
        # 2. Análisis de Estándares
        # ============================================
        estandares = df[df['Tipo'] == 'Estándar']
        rangos = {
            'Bajo': (0, 0.33),
            'Medio': (0.34, 0.66),
            'Alto': (0.67, 1.0)
        }
        with st.expander("🔬 Análisis de Estándares", expanded=True):
            if not estandares.empty:
                # ANOVA: Evaluar efecto del día y del analista
                model = ols('Absorbancia ~ C(Día_num) + C(Analista)', data=estandares).fit()
                anova_results = anova_lm(model, typ=2)
                st.markdown("#### Resultados ANOVA (Estándares)")
                st.write(anova_results)
                
                # Cálculo de RSD por rango de concentración
                rsd_results = []
                for nombre, (min_conc, max_conc) in rangos.items():
                    subset = estandares[(estandares['Concentración'] >= min_conc) & 
                                        (estandares['Concentración'] <= max_conc)]
                    if not subset.empty:
                        rsd_series = (subset.groupby('Día_num')['Absorbancia'].std(ddof=1) / 
                                      subset.groupby('Día_num')['Absorbancia'].mean()) * 100
                        rsd_results.append({
                            'Rango': nombre,
                            'RSD Promedio': rsd_series.mean(),
                            'RSD Máximo': rsd_series.max()
                        })
                rsd_df = pd.DataFrame(rsd_results)
                st.markdown("#### Resumen de RSD en Estándares por Rango de Concentración")
                st.dataframe(rsd_df)
                
                # Gráficos de Estándares
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(8,4))
                    sns.lineplot(data=estandares, x='Día_num', y='Absorbancia', 
                                 hue='Concentración', style='Analista',
                                 markers=True, dashes=False, ci=95, ax=ax1)
                    ax1.set_title("Tendencia de Absorbancia en Estándares")
                    ax1.set_xlabel("Días desde inicio")
                    ax1.set_ylabel("Absorbancia")
                    plt.tight_layout()
                    st.pyplot(fig1)
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(8,4))
                    sns.boxplot(data=rsd_df, x='Rango', y='RSD Máximo', palette='Set2', ax=ax2)
                    ax2.axhline(2, color='red', linestyle='--', label='Límite ICH (2%)')
                    ax2.set_title("RSD Máximo por Rango")
                    ax2.set_ylabel("RSD (%)")
                    ax2.legend()
                    plt.tight_layout()
                    st.pyplot(fig2)
                
                # Evaluar tendencia global (media diaria) mediante regresión lineal
                est_global = estandares.groupby('Día_num')['Absorbancia'].mean().reset_index()
                slope, intercept = np.polyfit(est_global['Día_num'], est_global['Absorbancia'], 1)
                st.metric("Pendiente de Tendencia (Estándares)", f"{slope:.2e}",
                          delta="Estable" if abs(slope) < 0.01 else "Inestable")
                
                # ============================================
                # Gráficos de Residuales en Estándares
                # ============================================
                st.markdown("#### Análisis de Residuales en Estándares")
                est_global['Estimado'] = slope * est_global['Día_num'] + intercept
                est_global['Residuo'] = est_global['Absorbancia'] - est_global['Estimado']
                
                col_res, col_violin = st.columns(2)
                with col_res:
                    fig_res, ax_res = plt.subplots(figsize=(8,4))
                    sns.scatterplot(data=est_global, x='Día_num', y='Residuo', ax=ax_res, color='blue')
                    ax_res.axhline(0, color='red', linestyle='--', label="Residuo = 0")
                    ax_res.set_title("Residuales (Scatter Plot)")
                    ax_res.set_xlabel("Día")
                    ax_res.set_ylabel("Residuo")
                    ax_res.legend()
                    plt.tight_layout()
                    st.pyplot(fig_res)
                with col_violin:
                    fig_violin, ax_violin = plt.subplots(figsize=(8,4))
                    sns.violinplot(x=est_global['Residuo'], color='lightgreen', ax=ax_violin)
                    ax_violin.set_title("Distribución de Residuales (Violin Plot)")
                    ax_violin.set_xlabel("Residuo")
                    plt.tight_layout()
                    st.pyplot(fig_violin)
            else:
                st.warning("No hay datos de estándares para análisis")
                
        # ============================================
        # 3. Análisis de Muestras (Concentración Teórica)
        # ============================================
        muestras = df[df['Tipo'] == 'Muestra']
        with st.expander("🧪 Análisis de Muestras", expanded=True):
            if not muestras.empty:
                rsd_results_muestras = []
                for nombre, (min_conc, max_conc) in rangos.items():
                    subset = muestras[(muestras['Concentración'] >= min_conc) & 
                                      (muestras['Concentración'] <= max_conc)]
                    if not subset.empty:
                        rsd_series = (subset.groupby('Día_num')['Absorbancia'].std(ddof=1) / 
                                      subset.groupby('Día_num')['Absorbancia'].mean()) * 100
                        rsd_results_muestras.append({
                            'Rango': nombre,
                            'RSD Promedio': rsd_series.mean(),
                            'RSD Máximo': rsd_series.max()
                        })
                rsd_muestras_df = pd.DataFrame(rsd_results_muestras)
                st.markdown("#### Resumen de RSD en Muestras por Rango Teórico")
                st.dataframe(rsd_muestras_df)
                
                # Gráfico de tendencia de Absorbancia en Muestras
                fig3, ax3 = plt.subplots(figsize=(10,4))
                sns.lineplot(data=muestras, x='Día_num', y='Absorbancia',
                             hue='Concentración', style='Analista',
                             markers=True, dashes=False, ci=95, ax=ax3)
                ax3.set_title("Tendencia de Absorbancia en Muestras")
                ax3.set_xlabel("Días desde inicio")
                plt.tight_layout()
                st.pyplot(fig3)
                
                # Evaluar estabilidad en Muestras mediante la pendiente de la tendencia
                muestras_global = muestras.groupby('Día_num')['Absorbancia'].mean().reset_index()
                slope_m, intercept_m = np.polyfit(muestras_global['Día_num'], muestras_global['Absorbancia'], 1)
                st.metric("Pendiente de Tendencia (Muestras)", f"{slope_m:.2e}",
                          delta="Estable" if abs(slope_m) < 0.01 else "Inestable")
            else:
                st.warning("No hay datos de muestras para análisis")
                
    except Exception as e:
        st.error(f"🚨 Error crítico: {str(e)}")

# Lógica principal para cada módulo
def procesar_archivo(datos, funcion_analisis):
    """Función mejorada para manejar diferentes codificaciones y formatos"""
    if datos is not None:
        try:
            # Determinar tipo de archivo
            if datos.name.endswith('.csv'):
                # Lista de codificaciones comunes a probar
                codificaciones = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                
                for encoding in codificaciones:
                    try:
                        datos.seek(0)  # Reiniciar el cursor del archivo
                        df = pd.read_csv(datos, encoding=encoding)
                        st.success(f"Archivo leído con codificación: {encoding}")
                        return funcion_analisis(df)
                    except UnicodeDecodeError:
                        continue
                
                # Si ninguna codificación funcionó
                st.error("Error de codificación. Pruebe guardar el archivo en UTF-8")
                return

            elif datos.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(datos)
                return funcion_analisis(df)

        except Exception as e:
            st.error(f"Error al procesar archivo: {str(e)}")
            st.error("Posibles soluciones:")
            st.markdown("""
                1. **Para archivos CSV:**
                   - Guarde el archivo con codificación UTF-8
                   - Evite caracteres especiales como ñ, acentos o símbolos
                   - Use Excel: *Archivo > Guardar Como > CSV UTF-8*
                
                2. **Para archivos Excel:**
                   - Verifique que no tenga fórmulas o formatos complejos
                   - Guarde como .xlsx (formato moderno)
            """)
        
if modulo == "Linealidad y Rango":

    st.header("Análisis de Linealidad y Rango")
    st.info("""
        **Datos requeridos para este módulo:**
        - **Concentración:** Concentraciones de las soluciones estándar.
        - **Absorbancia:** Valores de absorbancia medidos.
        - **Tipo:** Identificar si es "Estándar" o "Muestra".
        - **Día:** Identificar el día de la medición.""")  
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
    
    
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    procesar_archivo(datos, calcular_linealidad)
    

# Módulo de LOD y LOQ
elif modulo == "Límites de Detección y Cuantificación":
    st.header("Cálculo de LOD y LOQ")
    st.info("""
        **Datos requeridos para este módulo:**
        - **Concentración:** Concentraciones de las soluciones estándar.
        - **Absorbancia:** Valores de absorbancia medidos.
        - **Tipo:** Identificar si es "Estándar" o "Muestra".
        - **Día:** Día en que se realizó la medición.
        """)  
    
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    if datos:
        datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
        procesar_archivo(datos, calcular_lod_loq)
        graficar_curva_calibracion_streamlit(datos_df)

# Módulo de Precisión

elif modulo == "Precisión (Repetibilidad e Intermedia)":
    st.header("Evaluación de Precisión")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Absorbancia:** Datos de absorbancia agrupados por días y repeticiones.
        """
    )
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    procesar_archivo(datos, calcular_precision)

# Módulo de Exactitud
elif modulo == "Exactitud (Recuperación)":
    st.header("Cálculo de Exactitud")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Día:** Día en que se realizó la medición.
        - **Concentración Teórica:** Concentración fortificada conocida.
        - **Concentración Medida:** Concentración obtenida tras el análisis experimental.
        """
    )
    img_path = imagenes_dir / "conc_exac.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Concentración Teorica', 'Concentración Real', 'Día'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    procesar_archivo(datos, calcular_exactitud)

# Módulo de Robustez
elif modulo == "Robustez":
    st.header("Evaluación de Robustez")
    st.info("""
        **Datos requeridos para este módulo:**
        - **Factores variables:** Datos que representan condiciones variables del experimento.
        - **Resultados:** Datos de resultados obtenidos bajo dichas condiciones.
        """) 
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    procesar_archivo(datos, evaluar_robustez)


# Llamada a la función en el módulo de estabilidad
elif modulo == "Estabilidad":
    st.header("Evaluación de Estabilidad")
    st.info("""
        **Datos requeridos:**
        - Día | Concentración | Tipo | Absorbancia
        """)  
    st.image(str(imagenes_dir / "muestra.png"), caption="Estructura requerida")
    
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    procesar_archivo(datos, evaluar_estabilidad)  # Usa la función mejorada