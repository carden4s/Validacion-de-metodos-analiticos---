import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, f_oneway
import streamlit as st
from io import BytesIO
from pathlib import Path
from matplotlib.figure import Figure

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
        "Robustez"
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

def calcular_regresion(datos_dia):
    """Calcula la regresión lineal y los parámetros LOD y LOQ."""
    x, y = datos_dia['Concentración'], datos_dia['Absorbancia']
    if len(x) < 2 or len(y) < 2:
        st.error("No hay suficientes datos para realizar la regresión. Se requieren al menos 2 puntos.")
        return None, None, None, None, None
    regresion = linregress(x, y)
    slope, intercept = regresion.slope, regresion.intercept
    residuals = y - (slope * x + intercept)
    std_dev = residuals.std()
    lod = (3.3 * std_dev) / slope if slope != 0 else None
    loq = (10 * std_dev) / slope if slope != 0 else None
    return slope, intercept, lod, loq, std_dev


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

def calcular_precision(datos):
    """Evalúa la precisión siguiendo la guideline ICH Q2 mediante el cálculo del RSD (Relative Standard Deviation)."""
    columnas_necesarias = ['Día', 'Concentración', 'Absorbancia', 'Tipo']
    if not validar_columnas(datos, columnas_necesarias):
        return

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    datos_muestra = datos[datos['Tipo'] == 'Muestra']
    if datos_estandar.empty:
        st.error("No se encontraron datos de estándares en el conjunto de datos.")
        return

    datos_muestra['Concentración Real'] = np.nan
    for dia in datos['Día'].unique():
        estandares_dia = datos_estandar[datos_estandar['Día'] == dia]
        muestras_dia = datos_muestra[datos_muestra['Día'] == dia].copy()
        if estandares_dia.empty:
            st.warning(f"No se encontraron estándares para el día {dia}. Concentraciones no calculadas para este día.")
            continue

        try:
            X = estandares_dia['Absorbancia'].values.reshape(-1, 1)
            y = estandares_dia['Concentración'].values
            slope, intercept, _, _, _ = linregress(estandares_dia['Absorbancia'], estandares_dia['Concentración'])
            muestras_dia['Concentración Real'] = slope * muestras_dia['Absorbancia'] + intercept
            datos_muestra.update(muestras_dia)
            pendiente, intercepto = slope, intercept
            st.write(f"**Curva de calibración para el día {dia}:** Concentración = {pendiente:.4f} * Absorbancia + {intercepto:.4f}")
        except Exception as e:
            st.error(f"Error ajustando la curva de calibración para el día {dia}: {e}")

    for tipo, datos_tipo in [('Estándar', datos_estandar), ('Muestra', datos_muestra)]:
        st.subheader(f"**Precisión para tipo: {tipo}**")
        grupos_intraensayo = datos_tipo.groupby(['Día', 'Concentración'])['Absorbancia']
        rsd_intraensayo = grupos_intraensayo.std() / grupos_intraensayo.mean() * 100
        st.write("**RSD por día y concentración (Repetibilidad intraensayo):**")
        st.table(rsd_intraensayo.reset_index().rename(columns={'Absorbancia': 'RSD (%)'}))

        grupos_intermedio = datos_tipo.groupby('Concentración')['Absorbancia']
        rsd_intermedio = grupos_intermedio.std() / grupos_intermedio.mean() * 100
        st.write("**RSD por concentración (Precisión intermedia):**")
        st.table(rsd_intermedio.reset_index().rename(columns={'Absorbancia': 'RSD (%)'}))

        pesos = grupos_intermedio.mean()
        rsd_general = (rsd_intermedio * pesos / pesos.sum()).sum()
        st.write(f"**RSD General (Precisión total) para {tipo}:** {rsd_general:.2f}%")

        if rsd_general <= 3:
            st.success(f"{tipo}: Cumple con los criterios de precisión (RSD ≤ 3%).")
        else:
            st.error(f"{tipo}: No cumple con los criterios de precisión (RSD > 3%).")

    st.write("**Concentraciones reales calculadas para las muestras:**")
    st.table(datos_muestra[['Día', 'Absorbancia', 'Concentración Real']])

def calcular_exactitud(datos):
    """Calcula la exactitud con visualización profesional y validación ICH Q2."""
    # Configuración de estilo
    COLORS = ['#2ecc71', '#3498db', '#e74c3c']  # Verde, Azul, Rojo
    sns.set_theme(style="whitegrid", font_scale=0.95)
    
    # Validación mejorada
    columnas_necesarias = ['Día', 'Concentración', 'Absorbancia', 'Tipo']
    if not validar_columnas(datos, columnas_necesarias):
        return
    
    # Chequear tipos numéricos
    if not np.issubdtype(datos['Concentración'].dtype, np.number) or \
       not np.issubdtype(datos['Absorbancia'].dtype, np.number):
        st.error("❌ Las columnas 'Concentración' y 'Absorbancia' deben ser numéricas")
        return

    # Separar datos
    estandares = datos[datos['Tipo'] == 'Estándar']
    muestras = datos[datos['Tipo'] == 'Muestra']
    
    if estandares.empty:
        st.error("❌ No se encontraron datos de estándares para generar curvas")
        return

    with st.expander("📚 Método de Cálculo ICH Q2", expanded=True):
        st.markdown("""
        **Criterios de aceptación:**
        - Recuperación media: 98-102%
        - Desviación estándar relativa (RSD): ≤3%
        """)

    muestras_calculadas = []
    for dia in datos['Día'].unique():
        with st.container():
            st.markdown(f"## 📅 Día {dia}")
            est_dia = estandares[estandares['Día'] == dia]
            mues_dia = muestras[muestras['Día'] == dia]
            
            # Validación de datos
            if len(est_dia) < 3:
                st.warning(f"⚠️ Mínimo 3 estándares requeridos (Día {dia})")
                continue
                
            if mues_dia.empty:
                st.warning(f"⚠️ No hay muestras para el Día {dia}")
                continue
                
            try:
                # Cálculo de regresión
                regresion = linregress(est_dia['Absorbancia'], est_dia['Concentración'])
                slope, intercept = regresion.slope, regresion.intercept
                r_squared = regresion.rvalue**2
                
                # Cálculo de concentraciones
                mues_dia = mues_dia.copy()
                mues_dia['Concentración Medida'] = (slope * mues_dia['Absorbancia'] + intercept).round(4)
                mues_dia['Recuperación (%)'] = ((mues_dia['Concentración Medida'] / mues_dia['Concentración']) * 100).round(2)
                muestras_calculadas.append(mues_dia)

                # Mostrar métricas en columnas
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📈 Parámetros de la Curva")
                    st.metric("Ecuación", f"y = {slope:.4f}x + {intercept:.4f}")
                    st.metric("R²", f"{r_squared:.4f}")
                    
                with col2:
                    st.markdown("### 🔍 Control de Calidad")
                    st.metric("Número de Muestras", len(mues_dia))
                    recuperacion_prom = mues_dia['Recuperación (%)'].mean()
                    st.metric("Recuperación Promedio", 
                            f"{recuperacion_prom:.2f}%",
                            delta="ICH: 98-102%" if 98 <= recuperacion_prom <= 102 else None)

                # Gráfico de correlación
                with st.expander(f"📊 Relación Teórico vs Medido - Día {dia}", expanded=False):
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111)
                    sns.scatterplot(data=mues_dia, x='Concentración', y='Concentración Medida',
                                   s=100, color=COLORS[0], edgecolor='black')
                    lims = [mues_dia[['Concentración', 'Concentración Medida']].min().min(),
                           mues_dia[['Concentración', 'Concentración Medida']].max().max()]
                    ax.plot(lims, lims, '--', color=COLORS[2], alpha=0.5, label='Línea Ideal')
                    ax.set_title(f"Correlación de Concentraciones\nDía {dia}", fontsize=14)
                    ax.set_xlabel("Concentración Teórica (μg/mL)", fontsize=12)
                    ax.set_ylabel("Concentración Medida (μg/mL)", fontsize=12)
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    plt.close(fig)
                    
            except Exception as e:
                st.error(f"❌ Error en Día {dia}: {str(e)}")
                continue

    if not muestras_calculadas:
        st.error("❌ No se calcularon resultados válidos")
        return
        
    resultados = pd.concat(muestras_calculadas)
    
    # Análisis estadístico
    st.markdown("## 📌 Resumen de Exactitud")
    resumen = resultados.groupby('Día').agg(
        Muestras=('Recuperación (%)', 'size'),
        Media=('Recuperación (%)', lambda x: f"{x.mean():.2f}%"),
        DE=('Recuperación (%)', lambda x: f"{x.std():.2f}%"),
        Rango=('Recuperación (%)', lambda x: f"{x.min():.2f}% - {x.max():.2f}%")
    ).reset_index()
    
    # Estilo condicional
    def estilo_recuperacion(val):
        color = '#d4edda' if 98 <= float(val.strip('%')) <= 102 else '#f8d7da'
        return f'background-color: {color}'
    
    def estilo_rsd(val):
        color = '#d4edda' if float(val.strip('%')) <= 3 else '#f8d7da'
        return f'background-color: {color}'
    
    # Mostrar tabla interactiva
    st.dataframe(
        resumen.style
        .applymap(estilo_recuperacion, subset=['Media'])
        .applymap(estilo_rsd, subset=['DE'])
        .set_properties(**{'text-align': 'center'})
        .format(precision=2)
    )
    
    # Gráfico final de recuperación
    with st.expander("📈 Análisis Detallado de Recuperación", expanded=True):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        sns.boxplot(data=resultados, x='Día', y='Recuperación (%)', palette=COLORS)
        ax.axhline(98, color=COLORS[2], linestyle='--', alpha=0.5, label='Límite ICH Inferior')
        ax.axhline(102, color=COLORS[2], linestyle='--', alpha=0.5, label='Límite ICH Superior')
        ax.set_title("Distribución de Porcentajes de Recuperación", fontsize=14)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    
    # Generar descarga
    with st.expander("💾 Descargar Resultados Completos", expanded=False):
        generar_descarga(resultados)
        st.info("Los resultados incluyen todos los cálculos de concentración y recuperación")

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
    """Evalúa la robustez del método analítico mediante ANOVA."""
    columnas_necesarias = ['Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return

    factores_posibles = ['Día', 'Concentración', 'Tipo']
    factor = st.selectbox("Selecciona el factor a evaluar:", factores_posibles)
    if factor not in datos.columns:
        st.error(f"El factor '{factor}' no está en los datos.")
        return

    grupos = [grupo['Absorbancia'].values for _, grupo in datos.groupby(factor)]
    estadistico, p_value = f_oneway(*grupos)

    st.write(f"**Factor evaluado:** {factor}")
    st.write(f"**Estadístico F:** {estadistico:.4f}")
    st.write(f"**Valor p:** {p_value:.4e}")

    if p_value > 0.05:
        st.success("No hay diferencias significativas (p > 0.05). El método es robusto.")
    else:
        st.error("Hay diferencias significativas (p ≤ 0.05). El método no es robusto.")

    st.write("**Gráfico de caja (Boxplot):**")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=factor, y='Absorbancia', data=datos)
    plt.title(f"Variabilidad de Absorbancia según {factor}")
    st.pyplot(plt)

def evaluar_estabilidad(datos):
    """Evalúa la estabilidad de la solución en el tiempo."""
    columnas_necesarias = ['Tiempo', 'Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return

    x = datos['Tiempo']
    y = datos['Absorbancia']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    st.write(f"**Pendiente:** {slope:.4f}")
    st.write(f"**Intercepto:** {intercept:.4f}")
    st.write(f"**Coeficiente de determinación (R²):** {r_value**2:.4f}")

    if abs(slope) < 0.01:
        st.success("El método es estable en el tiempo (pendiente cercana a 0).")
    else:
        st.error("El método no es estable en el tiempo (pendiente alejada de 0).")

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=x, y=y, marker='o')
    plt.title("Estabilidad de la Solución")
    plt.xlabel("Tiempo")
    plt.ylabel("Absorbancia")
    st.pyplot(plt)

# Lógica principal para cada módulo
def procesar_archivo(datos, funcion_calculo):
    """Procesa el archivo subido y ejecuta la función de cálculo correspondiente."""
    if datos:
        try:
            datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
            previsualizar_datos(datos_df)
            funcion_calculo(datos_df)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

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
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Absorbancia', 'Tipo'")
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