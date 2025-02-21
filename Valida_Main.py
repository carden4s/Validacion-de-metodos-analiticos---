import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import BytesIO
from pathlib import Path
from datetime import datetime
from scipy.stats import linregress, f_oneway, t, ttest_1samp
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from reportlab.lib.pagesizes import letter, inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from sklearn.linear_model import LinearRegression
from patsy import dmatrices
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.stattools import durbin_watson 
from sklearn.cluster import KMeans
import pytz
import statsmodels.api as sm


# Obtener la ruta del directorio actual
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
imagenes_dir = current_dir / "img"

# Page Configuration
st.set_page_config(
    page_title="Validación De Métodos Analíticos",
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
        justify-content: space-between; /* Extend the title container across the full width */
        align-items: center;
        gap: 10px;
        margin-bottom: 20px; /* Spacing below title */
        width: 100%;
    }
    .title-container img:first-child {
        width: 120px; /* Adjust first icon size */
        height: auto;
    }
    .title-container h1 {
        flex: 1;
        text-align: center;
        margin: 0;
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
        <h1>Validación de Métodos Analíticos</h1>
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

def procesar_archivo(archivo, funcion_procesamiento, modulo):
    if archivo:
        try:
            # Inicializar generador de PDF
            pdf_gen = PDFGenerator(modulo)
            
            # Cargar datos
            if archivo.name.endswith('.csv'):
                data = pd.read_csv(archivo)
            else:
                data = pd.read_excel(archivo)
            
            # Verificar que funcion_procesamiento es callable o iterable de callables
            if isinstance(funcion_procesamiento, (list, tuple)):
                resultados = []
                for func in funcion_procesamiento:
                    if callable(func):
                        resultado = func(data, pdf_gen)
                        resultados.append(resultado)
                    else:
                        st.error("Error: Uno de los elementos de la lista/tuple de funciones no es callable.")
                        return False
                overall_result = all(resultados)
            else:
                if callable(funcion_procesamiento):
                    overall_result = funcion_procesamiento(data, pdf_gen)
                else:
                    st.error("Error: La función de procesamiento no es callable.")
                    return False
            
            # Generar PDF si el procesamiento es exitoso
            if overall_result:
                pdf = pdf_gen.generate_pdf()
                st.session_state['current_pdf'] = pdf
                st.session_state['current_module'] = modulo
                return True
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    return None


def calcular_linealidad(data, pdf_gen):
    """Calcula la linealidad y rango del método agrupando por Analista y Día."""
    sns.set_theme(style="whitegrid", palette="muted")
    COLORS = ['#2ecc71', '#e74c3c']
    
    # 1. Validación de columnas y datos
    req_cols = ['Concentración', 'Respuesta', 'Tipo', 'Día', 'Analista']
    if not validar_columnas(data, req_cols):
        return False

    data = data.copy()
    for col in ['Concentración', 'Respuesta']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=req_cols, inplace=True)

    # 2. Verificación de niveles de concentración
    niveles = data[data['Tipo'] == 'Estándar']['Concentración'].nunique()
    if niveles < 5:
        st.error("✘ Requiere 5 niveles de concentración (guía ICH Q2(R1))")
        return False

    cumplimiento_global = True
    metricas_modulo = []

    # 3. Análisis por Analista y Día
    for (analista, dia), grupo in data.groupby(['Analista', 'Día']):
        with st.container():
            st.markdown(f"## 🔬 Analista: {analista} | 📅 Día: {dia}")
            pdf_gen.add_subsection(f"Analista: {analista} - Día: {dia}")

            # 4. Procesamiento de estándares
            estandar = grupo[grupo['Tipo'] == 'Estándar']
            if estandar.empty:
                st.warning(f"⚠️ Sin estándares para {analista} - Día {dia}")
                continue

            # 5. Validación de triplicados
            conteo = estandar.groupby('Concentración').size()
            if any(conteo < 3):
                st.error(f"✘ Triplicados incompletos en {analista}-{dia} (USP <1225>)")
                continue

            # 6. Cálculos estadísticos
            try:
                estandar_prom = estandar.groupby('Concentración').agg(
                    Respuesta=('Respuesta', 'mean'),
                    DE=('Respuesta', 'std')
                ).reset_index()

                X = estandar_prom['Concentración']
                y = estandar_prom['Respuesta']
                reg = linregress(X, y)
                
                # Intervalo de confianza pendiente
                n = len(X)
                t_val = t.ppf(0.975, n-2)
                ci_slope = (reg.slope - t_val*reg.stderr, reg.slope + t_val*reg.stderr)
                slope_cumple = not (ci_slope[0] <= 0 <= ci_slope[1])
                
                # Residuales porcentuales
                pred_y = reg.slope*X + reg.intercept
                residual_pct = ((y - pred_y)/pred_y)*100
                residual_cumple = all(abs(residual_pct) <= 2)

            except Exception as e:
                st.error(f"Error en análisis: {str(e)}")
                continue

            # 7. Criterios de aceptación
            adj_r2 = 1 - (1 - reg.rvalue**2)*(n-1)/(n-2)
            cumple = all([adj_r2 >= 0.98, slope_cumple, residual_cumple])

            # 8. Registro de métricas
            metricas = {
                'Analista': analista,
                'Día': dia,
                'R²': adj_r2,
                'Pendiente': reg.slope,
                'IC Pendiente': f"[{ci_slope[0]:.4f}, {ci_slope[1]:.4f}]",
                'Residual Max (%)': f"{abs(residual_pct).max():.2f}",
                'Cumplimiento': cumple
            }
            metricas_modulo.append(metricas)

            # 9. Visualización y tablas
            fig = plt.figure(figsize=(14, 6), dpi=150)
            gs = fig.add_gridspec(2, 2)
            
            # Gráfico de regresión
            ax1 = fig.add_subplot(gs[0, 0])
            sns.regplot(x=X, y=y, ax=ax1, ci=95, 
                        scatter_kws={'s':80, 'edgecolor':'black'},
                        line_kws={'color':COLORS[0], 'lw':2})
            ax1.set_title(f"Regresión Lineal - {analista} (Día {dia})")
            
            # Residuales
            ax2 = fig.add_subplot(gs[0, 1])
            sns.scatterplot(x=X, y=residual_pct, ax=ax2, color=COLORS[1])
            ax2.axhline(0, color='black', ls='--')
            ax2.fill_between(X, -2, 2, color='gray', alpha=0.1)
            ax2.set_title("Residuales (%)")
            
            # Tabla para Streamlit
            ax3 = fig.add_subplot(gs[1, :])
            ax3.axis('off')
            tabla_streamlit = [
                ["Parámetro", "Valor", "Cumple"],
                [f"R² Ajustado", f"{adj_r2:.4f}", "✔" if adj_r2 >=0.98 else "✘"],
                [f"IC Pendiente", f"{ci_slope[0]:.4f}-{ci_slope[1]:.4f}", "✔" if slope_cumple else "✘"],
                [f"Residual Máx.", f"{abs(residual_pct).max():.2f}%", "✔" if residual_cumple else "✘"]
            ]
            tabla = ax3.table(
                cellText=tabla_streamlit, 
                loc='center', 
                colWidths=[0.3,0.4,0.3],
                cellLoc='center'
            )
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(12)
            
            # Estilizar celdas
            for row in [1,2,3]:
                celda = tabla.get_celld()[(row,2)]
                celda.set_text_props(
                    color='green' if "✔" in celda.get_text().get_text() else 'red',
                    weight='bold'
                )

            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabla para PDF
            pdf_gen.add_table(
                table_data=[
                    ["Parámetro", "Valor", "Cumplimiento"],
                    ["R² Ajustado", f"{adj_r2:.4f}", "Cumple" if adj_r2 >=0.98 else "No cumple"],
                    ["IC Pendiente", f"{ci_slope[0]:.4f} a {ci_slope[1]:.4f}", "Cumple" if slope_cumple else "No cumple"],
                    ["Residual Máximo", f"{abs(residual_pct).max():.2f}%", "Cumple" if residual_cumple else "No cumple"]
                ],
                title=f"Resultados {analista} - Día {dia}"
            )
            
            pdf_gen.capture_figure(fig, f"Linealidad_{analista}_{dia}")

    # 10. Resumen consolidado
    if metricas_modulo:
        df_metricas = pd.DataFrame(metricas_modulo)
        
        # Tabla resumen PDF
        tabla_resumen = [
            ["Analista", "Día", "R²", "IC Pendiente", "Cumplimiento"]
        ] + [
            [
                row['Analista'],
                str(row['Día']),
                f"{row['R²']:.4f}",
                row['IC Pendiente'],
                "Cumple" if row['Cumplimiento'] else "No cumple"
            ] for _, row in df_metricas.iterrows()
        ]
        
        pdf_gen.add_table(
            table_data=tabla_resumen,
            title="Resumen General de Linealidad"
        )

        # Conclusión final
        cumplimiento_global = all(df_metricas['Cumplimiento'])
        st.metric("Cumplimiento Global", 
                 "✅ Cumple" if cumplimiento_global else "❌ No Cumple",
                 delta=f"{df_metricas['Cumplimiento'].sum()}/{len(df_metricas)} grupos válidos")
        pdf_gen.add_metric("Cumplimiento Global", 
                          "Cumple" if cumplimiento_global else "No Cumple", 
                          cumplimiento_global)

    return True

def calcular_lod_loq(datos, pdf_gen):
    """Calcula LOD y LOQ con visualización y reporte PDF integrados según las guías de Farmacéuticos y CCAYAC."""
    COLORS = ['#3498db', '#2ecc71', '#e74c3c']  # Azul, Verde, Rojo
    sns.set_theme(style="whitegrid", font_scale=0.95)
    
    # Sección inicial en el PDF
    pdf_gen.add_section_title("Resultados de LOD y LOQ")
    pdf_gen.add_text_block("Este reporte presenta el cálculo de LOD (Límite de Detección) y LOQ (Límite de Cuantificación) "
                           "según la metodología validada por las guías de Farmacéuticos y CCAYAC.")
    
    # Validación de columnas y tipos numéricos
    columnas_necesarias = ['Día', 'Tipo', 'Concentración', 'Respuesta']
    if not validar_columnas(datos, columnas_necesarias):
        pdf_gen.add_text_block("✘ Error: Faltan columnas necesarias en el dataset.", style='error')
        return False

    if not np.issubdtype(datos['Concentración'].dtype, np.number) or \
       not np.issubdtype(datos['Respuesta'].dtype, np.number):
        pdf_gen.add_text_block("✘ Error: Las columnas 'Concentración' y 'Respuesta' deben ser numéricas.", style='error')
        st.error("✘ Las columnas 'Concentración' y 'Respuesta' deben ser numéricas")
        return False

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    if datos_estandar.empty:
        pdf_gen.add_text_block("✘ Error: No se encontraron datos de tipo 'Estándar'.", style='error')
        st.error("✘ No se encontraron datos de tipo 'Estándar'")
        return False

    # Mostrar la metodología en Streamlit y agregar versión "limpia" para el PDF
    with st.expander("📊 Método de Cálculo", expanded=True):
        contenido_markdown = """
        **Fórmulas aplicadas:**
        - $LOD = \\frac{3 \\times σ}{S}$  
        - $LOQ = \\frac{10 \\times σ}{S}$  
        Donde:
        - $σ$: Desviación estándar de los residuales (o blancos)  
        - $S$: Pendiente de la curva de calibración
        """
        st.markdown(contenido_markdown)
        contenido_pdf = (
            "Fórmulas aplicadas:\n"
            "- LOD = (3 × σ) / S\n"
            "- LOQ = (10 × σ) / S\n"
            "Donde:\n"
            "- σ: Desviación estándar de los residuales (o blancos)\n"
            "- S: Pendiente de la curva de calibración"
        )
        pdf_gen.add_subsection("Metodología de Cálculo")
        pdf_gen.add_text_block(contenido_pdf)
    
    # Procesar cada día
    for dia in datos_estandar['Día'].unique():
        with st.container():
            st.markdown(f"## 📅 Día {dia}")
        
            datos_dia = datos_estandar[datos_estandar['Día'] == dia]
            if len(datos_dia) < 3:
                st.warning(f"⚠️ Mínimo 3 puntos requeridos para un cálculo confiable (Día {dia})")
                pdf_gen.add_text_block(f"⚠️ Mínimo 3 puntos requeridos para un cálculo confiable (Día {dia})", style='error')
                continue
            try:
                # Cálculos de regresión y parámetros
                regresion = linregress(datos_dia['Concentración'], datos_dia['Respuesta'])
                slope = regresion.slope
                intercept = regresion.intercept
                r_value = regresion.rvalue
                r_squared = r_value**2
                residuals = datos_dia['Respuesta'] - (slope * datos_dia['Concentración'] + intercept)
                std_dev = residuals.std()
                # Uso de factores 3 y 10 según las guías
                lod = (3 * std_dev) / slope if slope != 0 else np.nan
                loq = (10 * std_dev) / slope if slope != 0 else np.nan
                
                # Nota: Los criterios de aceptación requieren evaluar que:
                # - LOD < especificación de impurezas
                # - LOQ ≤ 50% del límite de especificación
                # Dado que estas especificaciones dependen de la matriz, se deben comparar externamente.
                
                # Mostrar métricas en Streamlit (se usan tres columnas)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pendiente (S)", f"{slope:.4f}")
                    st.metric("R²", f"{r_squared:.4f}")
                with col2:
                    st.metric("LOD (μg/mL)", f"{lod:.4f}", help="(3×σ)/S")
                    st.metric("LOQ (μg/mL)", f"{loq:.4f}", help="(10×σ)/S")
                
                # Agregar métricas al PDF (tabla)
                metricas_dia = {
                    'Pendiente (S)': f"{slope:.4f}",
                    'R²': f"{r_squared:.4f}",
                    'LOD (μg/mL)': f"{lod:.4f}",
                    'LOQ (μg/mL)': f"{loq:.4f}"
                }
                pdf_gen.add_metrics_table(f"Métricas Día {dia}", metricas_dia)
                
                # Crear gráfica: mostrar línea de regresión y líneas verticales en LOD y LOQ (si están en rango)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(datos_dia['Concentración'], datos_dia['Respuesta'], color='lightgray', label='Datos estándar')
                x_vals = np.linspace(datos_dia['Concentración'].min(), datos_dia['Concentración'].max(), 100)
                y_vals = slope * x_vals + intercept
                ax.plot(x_vals, y_vals, color=COLORS[0], label='Línea de regresión')
                if datos_dia['Concentración'].min() <= lod <= datos_dia['Concentración'].max():
                    ax.axvline(lod, color=COLORS[2], linestyle='--', label=f'LOD = {lod:.2f}')
                if datos_dia['Concentración'].min() <= loq <= datos_dia['Concentración'].max():
                    ax.axvline(loq, color=COLORS[1], linestyle='-.', label=f'LOQ = {loq:.2f}')
                ax.set_title(f"Calibración y Límites - Día {dia} (R² = {r_squared:.4f})", fontsize=14)
                ax.set_xlabel("Concentración (μg/mL)", fontsize=12)
                ax.set_ylabel("Respuesta (UA)", fontsize=12)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
                pdf_gen.capture_figure(fig, f"Limites_Dia_{dia}")
                plt.close(fig)
                
            except Exception as e:
                st.error(f"✘ Error en Día {dia}: {str(e)}")
                pdf_gen.add_text_block(f"✘ Error en Día {dia}: {str(e)}", style='error')
                continue
    
    return True

def graficar_curva_calibracion_streamlit(datos, pdf_gen):
    """Grafica la curva de calibración con estilo profesional e integra la gráfica y la tabla en el reporte PDF, según las guías de Farmacéuticos y CCAYAC."""
    COLORS = ['#2ecc71', '#3498db', '#e74c3c']  # Verde, Azul, Rojo
    plt.style.use('seaborn-v0_8-talk')
    
    # Validación de datos
    columnas_necesarias = ['Día', 'Tipo', 'Concentración', 'Respuesta']
    if not validar_columnas(datos, columnas_necesarias):
        pdf_gen.add_text_block("✘ Error: Faltan columnas necesarias en el dataset para calibración.", style='error')
        return False

    datos_estandar = datos[datos['Tipo'] == 'Estándar']
    if datos_estandar.empty:
        pdf_gen.add_text_block("✘ Error: No se encontraron datos de calibración.", style='error')
        st.error("✘ No se encontraron datos de calibración")
        return False

    pdf_gen.add_section_title("Curvas de Calibración")
    
    for dia in datos_estandar['Día'].unique():
        with st.container():
            st.markdown(f"## 📈 Curva de Calibración - Día {dia}")
            datos_dia = datos_estandar[datos_estandar['Día'] == dia]
            if len(datos_dia) < 3:
                st.warning(f"⚠️ Mínimo 3 puntos recomendados para curva confiable (Día {dia})")
                pdf_gen.add_text_block(f"⚠️ Mínimo 3 puntos recomendados para curva confiable (Día {dia})", style='error')
                continue
            try:
                # Cálculos de regresión y parámetros
                regresion = linregress(datos_dia['Concentración'], datos_dia['Respuesta'])
                slope = regresion.slope
                intercept = regresion.intercept
                r_value = regresion.rvalue
                r_squared = r_value**2
                residuals = datos_dia['Respuesta'] - (slope * datos_dia['Concentración'] + intercept)
                std_dev = residuals.std()
                # Uso de los factores 3 y 10 según la metodología
                lod = (3 * std_dev) / slope if slope != 0 else np.nan
                loq = (10 * std_dev) / slope if slope != 0 else np.nan

                # Crear gráfica profesional de calibración
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')
                sns.regplot(x=datos_dia['Concentración'], y=datos_dia['Respuesta'],
                           ax=ax, ci=95,
                           scatter_kws={'s': 80, 'color': COLORS[0], 'edgecolor': 'black', 'alpha': 0.8},
                           line_kws={'color': COLORS[1], 'lw': 2, 'alpha': 0.8})
                ax.axvline(lod, color=COLORS[2], linestyle='--', lw=2, alpha=0.8, label=f'LOD ({lod:.2f})')
                ax.axvline(loq, color=COLORS[1], linestyle='-.', lw=2, alpha=0.8, label=f'LOQ ({loq:.2f})')
                textstr = '\n'.join((
                    f'$R^2 = {r_squared:.4f}$',
                    f'$y = {slope:.4f}x + {intercept:.4f}$',
                    f'$σ = {std_dev:.4f}$'
                ))
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.set_title(f"Curva de Calibración - Día {dia}", fontsize=16, pad=20)
                ax.set_xlabel("Concentración (μg/mL)", fontsize=14)
                ax.set_ylabel("Respuesta (UA)", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='lower right', frameon=True, shadow=True)
                # Opcional: resaltar la zona comprendida entre 0 y LOQ
                ax.axvspan(0, loq, facecolor='#f8d7da', alpha=0.3, label='Zona LOD/LOQ')
                st.pyplot(fig)
                pdf_gen.capture_figure(fig, f"Calibracion_Dia_{dia}")
                plt.close(fig)

                # Mostrar tabla de datos en Streamlit
                with st.expander(f"📋 Datos de Calibración - Día {dia}"):
                    st.dataframe(
                        datos_dia[['Concentración', 'Respuesta']]
                        .style.format("{:.4f}")
                    )
                    
            except Exception as e:
                st.error(f"✘ Error en Día {dia}: {str(e)}")
                pdf_gen.add_text_block(f"✘ Error en Día {dia}: {str(e)}", style='error')
                continue

    return True

def agrupar_valores(valores, umbral=1.0):
    """
    Agrupa una lista ordenada de valores en grupos donde la diferencia entre elementos consecutivos es <= umbral.
    Devuelve una lista de grupos (cada grupo es una lista de valores).
    """
    grupos = []
    if len(valores) == 0:
        return grupos
    grupo_actual = [valores[0]]
    for v in valores[1:]:
        if v - grupo_actual[-1] <= umbral:
            grupo_actual.append(v)
        else:
            grupos.append(grupo_actual)
            grupo_actual = [v]
    grupos.append(grupo_actual)
    return grupos

def calcular_precision_por_rango(datos, pdf_gen, umbral_agrupacion=1.0):
    """
    Realiza el análisis de precisión (Precisión del Sistema e Intermedia) para cada rango de concentración,
    agrupando automáticamente los valores de 'Concentración' según una tolerancia de ±umbral_agrupacion.
    
    Para la Precisión del Sistema (8.1), se calcula el CV para cada grupo (Analista, Día) en ese rango y se
    selecciona el grupo con el menor CV para reportarlo.
    
    Para la Precisión Intermedia (8.7), se calcula el CV por cada combinación (Analista, Día) y se resumen los
    valores (por ejemplo, se reporta el CV mínimo y el promedio de CV de los grupos).
    
    Se espera que 'datos' tenga las columnas:
       ['Concentración', 'Respuesta', 'Día', 'Analista', 'Tipo']
    
    Los resultados se muestran en Streamlit y se agregan al PDF mediante 'pdf_gen'.
    """
    st.header("🎯 Análisis de Precisión por Rango (Agrupamiento Automático)")
    pdf_gen.add_section_title("Análisis de Precisión por Rango (Agrupamiento Automático)")

    # 1. Validar columnas requeridas
    required_cols = ['Concentración', 'Respuesta', 'Día', 'Analista', 'Tipo']
    for col in required_cols:
        if col not in datos.columns:
            st.error(f"Falta la columna '{col}' en el DataFrame.")
            pdf_gen.add_text_block(f"✘ Error: Falta la columna '{col}' en el DataFrame.", style="error")
            return False

    # 2. Conversión numérica y limpieza
    for col in ['Concentración', 'Respuesta']:
        datos[col] = pd.to_numeric(datos[col], errors='coerce')
    datos.dropna(subset=['Concentración', 'Respuesta'], inplace=True)
    datos['Tipo'] = datos['Tipo'].str.lower()

    # 3. Agrupar automáticamente los valores de concentración con tolerancia ±umbral_agrupacion
    unique_vals = sorted(datos['Concentración'].unique())
    grupos_conc = agrupar_valores(unique_vals, umbral=umbral_agrupacion)
    mapping = {}
    for grupo in grupos_conc:
        rep = np.median(grupo)
        for v in grupo:
            mapping[v] = rep
    datos['Rango'] = datos['Concentración'].map(mapping).apply(lambda x: f"{x:.2f}")

    # Mostrar la agrupación para depuración
    st.subheader("Agrupación de Concentraciones")
    df_grupos = datos[['Concentración', 'Rango']].drop_duplicates().sort_values(by='Concentración')
    st.dataframe(df_grupos)
    tabla_grupos = [["Concentración", "Rango asignado"]]
    for _, row in df_grupos.iterrows():
        tabla_grupos.append([f"{row['Concentración']:.2f}", row['Rango']])
    pdf_gen.add_table(tabla_grupos, title="Agrupación de Concentraciones (Automática)")

    # 4. Iterar sobre cada rango disponible
    rangos_disponibles = sorted(datos['Rango'].unique(), key=lambda x: float(x))
    for rango_label in rangos_disponibles:
        df_rango = datos[datos['Rango'] == rango_label].copy()
        if df_rango.empty:
            continue

        st.markdown(f"### Análisis para el Rango {rango_label}")
        pdf_gen.add_subsection(f"Rango {rango_label}")

        # Filtrar estándares
        estandares = df_rango[df_rango['Tipo'] == 'estándar'].copy()
        if estandares.empty:
            st.warning(f"No se encontraron datos de tipo 'estándar' en el rango {rango_label}.")
            pdf_gen.add_text_block(f"⚠️ No se encontraron datos de tipo 'estándar' en el rango {rango_label}.", style="warning")
            continue

        # ------------------------------------------------
        # A: Precisión del Sistema (8.1)
        # ------------------------------------------------
        with st.expander(f"📊 Precisión del Sistema - Rango {rango_label}", expanded=False):
            pdf_gen.add_subsection(f"Precisión del Sistema (8.1) - Rango {rango_label}")
            grupos = estandares.groupby(['Analista', 'Día'])
            resultados = []
            for (analista, dia), grupo in grupos:
                if len(grupo) < 6:
                    st.warning(f"Analista {analista} (Día {dia}): Insuficientes réplicas ({len(grupo)}/6)")
                    continue
                cv = (grupo['Respuesta'].std() / grupo['Respuesta'].mean()) * 100
                resultados.append({
                    'Analista': analista,
                    'Día': dia,
                    'Réplicas': len(grupo),
                    'CV (%)': cv,
                    'Media': grupo['Respuesta'].mean(),
                    'Std': grupo['Respuesta'].std()
                })
            if not resultados:
                st.error(f"Rango {rango_label}: No hay grupos válidos (Analista-Día) con ≥6 réplicas para la precisión del sistema.")
                pdf_gen.add_text_block(f"✘ Rango {rango_label}: Insuficientes grupos (Analista-Día) para calcular precisión del sistema.", style="error")
            else:
                df_resultados = pd.DataFrame(resultados)
                mejor_grupo = df_resultados.loc[df_resultados['CV (%)'].idxmin()]
                st.markdown(f"**Mejor grupo:** Analista {mejor_grupo['Analista']} - Día {mejor_grupo['Día']} con CV = {mejor_grupo['CV (%)']:.2f}%")
                pdf_gen.add_text_block(f"Mejor grupo (Precisión del Sistema) en el rango {rango_label}: "
                                       f"Analista {mejor_grupo['Analista']} - Día {mejor_grupo['Día']} con CV = {mejor_grupo['CV (%)']:.2f}%.")
                st.dataframe(df_resultados.style.format({"CV (%)": "{:.2f}"}), use_container_width=True)

                metodo_sistema = st.selectbox(f"Tipo de Método para Precisión del Sistema (rango {rango_label}):",
                                              ["fisico_quimico", "biologico"], key=f"sistema_{rango_label}")
                umbral_sistema = 1.5 if metodo_sistema == "fisico_quimico" else 3.0

                st.metric("Media Respuesta (Mejor grupo)", f"{mejor_grupo['Media']:.4f}")
                st.metric("Desv. Estándar (Mejor grupo)", f"{mejor_grupo['Std']:.4f}")
                st.metric("CV (Mejor grupo)", f"{mejor_grupo['CV (%)']:.2f}%", 
                          delta="Cumple" if mejor_grupo['CV (%)'] <= umbral_sistema else "No Cumple")

                grupo_mejor = grupos.get_group((mejor_grupo['Analista'], mejor_grupo['Día']))
                fig_sistema = plt.figure(figsize=(6, 4))
                sns.scatterplot(data=grupo_mejor, x='Concentración', y='Respuesta', hue='Día', style='Analista')
                plt.title(f"Precisión del Sistema - Mejor grupo (Rango {rango_label})")
                plt.axhline(mejor_grupo['Media'], color='red', linestyle='--', label='Media')
                plt.legend()
                st.pyplot(fig_sistema)
                pdf_gen.capture_figure(fig_sistema, f"dispersion_sistema_{rango_label}_mejor")
                plt.close(fig_sistema)

                tabla_sistema = [
                    ["Parámetro", "Valor"],
                    ["Analista", mejor_grupo['Analista']],
                    ["Día", mejor_grupo['Día']],
                    ["Réplicas", f"{mejor_grupo['Réplicas']}"],
                    ["Media Respuesta", f"{mejor_grupo['Media']:.4f}"],
                    ["Desv. Estándar", f"{mejor_grupo['Std']:.4f}"],
                    ["CV Sistema", f"{mejor_grupo['CV (%)']:.2f}%"],
                    ["Umbral", f"{umbral_sistema}%"],
                    ["Cumplimiento", "Cumple" if mejor_grupo['CV (%)'] <= umbral_sistema else "No Cumple"]
                ]
                pdf_gen.add_table(tabla_sistema, title=f"Precisión del Sistema - Rango {rango_label}")

        # ------------------------------------------------
        # B: Precisión Intermedia (8.7)
        # ------------------------------------------------
        with st.expander(f"📈 Precisión Intermedia - Rango {rango_label}", expanded=False):
            pdf_gen.add_subsection(f"Precisión Intermedia (8.7) - Rango {rango_label}")
            if estandares['Día'].nunique() < 2 or estandares['Analista'].nunique() < 2:
                st.error(f"Rango {rango_label}: Se requieren datos de al menos 2 días y 2 analistas para la precisión intermedia.")
                pdf_gen.add_text_block(f"✘ Rango {rango_label}: Insuficientes días o analistas para precisión intermedia.", style="error")
            else:
                cv_inter_df = (estandares.groupby(['Día', 'Analista'])['Respuesta']
                               .apply(lambda x: (x.std() / x.mean()) * 100)
                               .reset_index(name='CV'))
                cv_min = cv_inter_df['CV'].min()
                cv_mean = cv_inter_df['CV'].mean()
                
                st.markdown("#### CV por Día y Analista")
                st.dataframe(cv_inter_df)
                fig_inter = plt.figure(figsize=(6, 4))
                sns.barplot(data=cv_inter_df, x='Día', y='CV', hue='Analista')
                plt.axhline(cv_min, color='gray', linestyle=':', label=f"CV Mínimo = {cv_min:.2f}%")
                plt.axhline(cv_mean, color='blue', linestyle=':', label=f"CV Promedio = {cv_mean:.2f}%")
                plt.title(f"Precisión Intermedia - Rango {rango_label}")
                plt.legend()
                st.pyplot(fig_inter)
                pdf_gen.capture_figure(fig_inter, f"cv_inter_{rango_label}")
                plt.close(fig_inter)

                # En lugar de un CV global calculado sobre todos los datos,
                # se reportan los valores resumen (mínimo y promedio) de CV por grupo.
                metodo_inter = st.selectbox(f"Tipo de Método para Precisión Intermedia (rango {rango_label}):",
                                            ["cromatografico", "quimico", "espectrofotometrico", "biologico"],
                                            key=f"inter_{rango_label}")
                if metodo_inter == "cromatografico":
                    umbral_inter = 2.0
                elif metodo_inter in ["quimico", "espectrofotometrico"]:
                    umbral_inter = 3.0
                elif metodo_inter == "biologico":
                    umbral_inter = 5.0
                else:
                    umbral_inter = 2.0

                st.metric("CV Mínimo (Día-Analista)", f"{cv_min:.2f}%", 
                          delta="Cumple" if cv_min <= umbral_inter else "No Cumple")
                st.metric("CV Promedio (Rango)", f"{cv_mean:.2f}%", 
                          delta="Cumple" if cv_mean <= umbral_inter else "No Cumple")

                tabla_inter = [
                    ["Parámetro", "Valor"],
                    ["CV Mínimo", f"{cv_min:.2f}%"],
                    ["CV Promedio", f"{cv_mean:.2f}%"],
                    ["Umbral Intermedio", f"{umbral_inter}%"]
                ]
                pdf_gen.add_table(tabla_inter, title=f"Precisión Intermedia - Rango {rango_label}")

    st.success("✔ Análisis de Precisión por Rango finalizado para todos los rangos.")
    pdf_gen.add_text_block("✔ Análisis de Precisión por Rango finalizado para todos los rangos.")
    return True

#####################################
# Clase PDFGenerator
#####################################
class PDFGenerator:
    def __init__(self, modulo):
        self.modulo = modulo
        self.buffers = []           # Para imágenes (figuras)
        self.metrics = []           # Para métricas individuales (opcional)
        self.metrics_tables = []    # Para almacenar tablas de métricas (diccionarios)
        self.custom_tables = []     # Para tablas personalizadas (listas de listas)
        self.text_blocks = []       # Para bloques de texto
        self.sections = []          # Para títulos de sección
        self.styles = getSampleStyleSheet()
        if 'error' not in self.styles.byName:
            self.styles.add(ParagraphStyle(name='error', parent=self.styles['Normal'], textColor=colors.red))
        if 'conclusion' not in self.styles.byName:
            self.styles.add(ParagraphStyle(name='conclusion', parent=self.styles['Normal'], textColor=colors.green))

    def add_table(self, table_data, title=None, headers=None):
        """Versión corregida que maneja headers personalizados"""
        if headers:
            # Insertar headers como primera fila
            table_data = [headers] + table_data
        self.custom_tables.append((title, table_data))

    def capture_figure(self, fig, fig_name=None):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        self.buffers.append(buf)
        plt.close(fig)

    def add_metric(self, label, value, cumplimiento):
        self.metrics.append({
            'label': label,
            'value': value,
            'cumplimiento': cumplimiento
        })

    def add_metrics_table(self, title, metrics):
        self.metrics_tables.append((title, metrics))

    def add_subsection(self, title):
        subsection_style = ParagraphStyle(
            name='SubsectionStyle',
            parent=self.styles['Heading3'],
            fontSize=11,
            textColor=colors.darkblue,
            spaceAfter=8,
            alignment=0
        )
        self.text_blocks.append((title, subsection_style))

    def add_section_title(self, title):
        self.sections.append(title)

    def add_text_block(self, text, style=None):
        if isinstance(style, str):
            style = self.styles.get(style, self.styles['Normal'])
        elif style is None or not hasattr(style, "name"):
            style = self.styles['Normal']
        self.text_blocks.append((text, style))

    def generate_pdf(self):
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        elements = []
        
        title_style = ParagraphStyle(
            name='TitleStyle',
            parent=self.styles['Title'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=20,
            alignment=1
        )
        section_style = ParagraphStyle(
            name='SectionStyle',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.darkblue,
            spaceAfter=10,
            alignment=0
        )
        
        # Portada
        mexico_tz = pytz.timezone("America/Mexico_City")
        elements.append(Paragraph(f"Reporte de Validación: {self.modulo}", title_style))
        elements.append(Paragraph(f"Fecha: {datetime.now(mexico_tz).strftime('%d/%m/%Y %H:%M')}", self.styles['Normal']))
        elements.append(Spacer(1, 30))
        
        for sec in self.sections:
            elements.append(Paragraph(sec, section_style))
            elements.append(Spacer(1, 15))
        
        for text, style in self.text_blocks:
            elements.append(Paragraph(text, style))
            elements.append(Spacer(1, 12))
        
        for buf in self.buffers:
            img = RLImage(buf, width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 15))
        
        if self.metrics_tables:
            for title, met_dict in self.metrics_tables:
                elements.append(Paragraph(title, self.styles['Heading2']))
                elements.append(Spacer(1, 12))
                table_data = [["Métrica", "Valor"]]
                for key, value in met_dict.items():
                    table_data.append([key, value])
                tabla = Table(table_data, colWidths=[3*inch, 3*inch])
                tabla.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                elements.append(tabla)
                elements.append(Spacer(1, 20))
        
        if self.custom_tables:
            for title, table_data in self.custom_tables:
                if title:
                    elements.append(Paragraph(title, self.styles['Heading2']))
                    elements.append(Spacer(1, 12))
                num_cols = len(table_data[0])
                col_width = 6*inch / num_cols
                tabla = Table(table_data, colWidths=[col_width]*num_cols)
                tabla.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                elements.append(tabla)
                elements.append(Spacer(1, 20))
        
        if self.metrics:
            elements.append(Paragraph("Métricas Clave", self.styles['Heading2']))
            metric_data = [["Parámetro", "Valor", "Cumplimiento"]]
            for metric in self.metrics:
                metric_data.append([
                    metric['label'],
                    metric['value'],
                    "✔" if metric['cumplimiento'] else "✘"
                ])
            metric_table = Table(metric_data, colWidths=[3*inch, 2*inch, 1.5*inch])
            metric_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(metric_table)
            elements.append(Spacer(1, 20))
        
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer

#########################
# Función procesar_archivo
#########################
def procesar_archivo(archivo, funcion_procesamiento, modulo):
    if archivo:
        try:
            # Inicializar generador de PDF
            pdf_gen = PDFGenerator(modulo)
            
            # Cargar datos
            if archivo.name.endswith('.csv'):
                data = pd.read_csv(archivo)
            else:
                data = pd.read_excel(archivo)
            
            # Verificar que funcion_procesamiento es callable o iterable de callables
            if isinstance(funcion_procesamiento, (list, tuple)):
                resultados = []
                for func in funcion_procesamiento:
                    if callable(func):
                        resultado = func(data, pdf_gen)
                        resultados.append(resultado)
                    else:
                        st.error("Error: Uno de los elementos de la lista/tuple de funciones no es callable.")
                        return False
                overall_result = all(resultados)
            else:
                if callable(funcion_procesamiento):
                    overall_result = funcion_procesamiento(data, pdf_gen)
                else:
                    st.error("Error: La función de procesamiento no es callable.")
                    return False
            
            # Generar PDF si el procesamiento es exitoso
            if overall_result:
                pdf = pdf_gen.generate_pdf()
                st.session_state['current_pdf'] = pdf
                st.session_state['current_module'] = modulo
                return True
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    return None

def calcular_exactitud(datos, pdf_gen):
    """
    Calcula la exactitud mediante recuperación según las guías de Farmacéuticos y CCAYAC y genera un reporte PDF integrado.
    Se incluye:
      - Cálculo de la recuperación (%) redondeada a dos decimales.
      - Estadísticos descriptivos: media, DE, mediana, Q1, Q3, IQR.
      - Intervalo de confianza (IC) al 95% para la media.
      - Prueba t (H0: μ = 100) con p-valor.
      - Detección de outliers (método IQR).
      - Visualizaciones: boxplot con límites de aceptación, histograma, gráfico de tendencia y datos detallados.
    """
    # Validar columnas requeridas
    columnas_necesarias = ['Día', 'Concentración Teórica', 'Concentración Real']
    if not validar_columnas(datos, columnas_necesarias):
        pdf_gen.add_text_block("✘ Error: Faltan columnas necesarias en el dataset.", style="error")
        return False

    # Calcular porcentaje de recuperación redondeado a dos decimales
    datos['Recuperación (%)'] = ((datos['Concentración Real'] / datos['Concentración Teórica']) * 100).round(2)

    # Detección de outliers usando el método IQR para cada día
    def detectar_outliers(x):
        Q1 = x.quantile(0.25)
        Q3 = x.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ~((x >= lower_bound) & (x <= upper_bound))
    datos['Outlier'] = datos.groupby('Día')['Recuperación (%)'].transform(detectar_outliers)
    
    # Sección inicial del PDF
    pdf_gen.add_section_title("Análisis de Exactitud - Guía Farmacéuticos y CCAYAC")
    pdf_gen.add_text_block(
        "Este reporte presenta el análisis de exactitud mediante recuperación, comparando la concentración teórica "
        "versus la concentración real. Se evalúan los siguientes criterios:\n"
        "- Recuperación media entre 98-102%\n"
        "- Desviación estándar (DE) ≤ 3%\n"
        "- Prueba t (H0: μ = 100) sin significación (p > 0.05)"
    )
    
    # Análisis estadístico agrupado por día
    resumen_list = []
    dias = sorted(datos['Día'].unique())
    for dia in dias:
        grupo = datos[datos['Día'] == dia]
        n = len(grupo)
        media = grupo['Recuperación (%)'].mean()
        de = grupo['Recuperación (%)'].std()
        mediana = grupo['Recuperación (%)'].median()
        Q1 = grupo['Recuperación (%)'].quantile(0.25)
        Q3 = grupo['Recuperación (%)'].quantile(0.75)
        # Calcular intervalo de confianza al 95% para la media (si n > 1)
        if n > 1:
            t_crit = t.ppf(0.975, df=n-1)
            ci_lower = media - t_crit * (de/np.sqrt(n))
            ci_upper = media + t_crit * (de/np.sqrt(n))
        else:
            ci_lower, ci_upper = np.nan, np.nan
        # Prueba t de una muestra: H0: μ = 100
        t_stat, p_val = ttest_1samp(grupo['Recuperación (%)'], 100)
        # Evaluación de criterios de aceptación
        cumple_media = 98 <= media <= 102
        cumple_de = de <= 3
        cumplimiento = "✔" if (cumple_media and cumple_de) else "✘"
        
        resumen_list.append({
            'Día': dia,
            'Muestras': n,
            'Media (%)': media,
            'DE (%)': de,
            'Mediana (%)': mediana,
            'Q1 (%)': Q1,
            'Q3 (%)': Q3,
            'IC Media (%)': f"[{ci_lower:.2f}%, {ci_upper:.2f}%]" if n > 1 else "N/A",
            'p-valor': p_val,
            'Cumplimiento': cumplimiento
        })
    resumen = pd.DataFrame(resumen_list)
    
    # Mostrar resultados en la app Streamlit en pestañas
    st.header("Análisis de Exactitud - Guía Farmacéuticos y CCAYAC")
    tab1, tab2, tab3, tab4 = st.tabs(["Resumen Estadístico", "Distribución de Recuperaciones", "Tendencia", "Datos Detallados"])
    
    # --- Pestaña 1: Resumen Estadístico ---
    with tab1:
        st.subheader("Estadísticos por Día")
        st.dataframe(
            resumen.style.apply(lambda row: ['background: #e6f4ea' if row['Cumplimiento'] == '✔' else 'background: #fce8e6']*len(row), axis=1)
        )
        st.markdown("""
        **Criterios de Aceptación:**
        - Recuperación media entre 98-102%
        - Desviación estándar (DE) ≤ 3%
        - Prueba t (H0: μ = 100) sin significación (p > 0.05)
        """)
        
        # Dividir el resumen en dos tablas:
        # Tabla 1: Datos generales
        tabla_general_columns = ['Día', 'Muestras', 'Media (%)', 'DE (%)', 'Cumplimiento']
        tabla_detalle_columns = ['Día', 'Mediana (%)', 'Q1 (%)', 'Q3 (%)', 'IC Media (%)', 'p-valor']
        
        tabla_general = []
        tabla_detalle = []
        
        # Formatear encabezados con ancho fijo (15 caracteres)
        encabezado_general = [f"{col:<15}" for col in tabla_general_columns]
        encabezado_detalle = [f"{col:<15}" for col in tabla_detalle_columns]
        tabla_general.append(encabezado_general)
        tabla_detalle.append(encabezado_detalle)
        
        # Recorrer cada fila del resumen para agregar a las dos tablas
        for _, row in resumen.iterrows():
            fila_general = []
            fila_detalle = []
            for col in tabla_general_columns:
                valor = row[col]
                if isinstance(valor, (int, float)):
                    fila_general.append(f"{valor:<15.2f}")
                else:
                    fila_general.append(f"{str(valor):<15}")
            for col in tabla_detalle_columns:
                valor = row[col]
                # Para los valores numéricos en detalle, se formatean a 2 decimales
                if isinstance(valor, (int, float)):
                    fila_detalle.append(f"{valor:<15.2f}")
                else:
                    fila_detalle.append(f"{str(valor):<15}")
            tabla_general.append(fila_general)
            tabla_detalle.append(fila_detalle)
        
        # Agregar ambas tablas al PDF con títulos distintos
        pdf_gen.add_table(tabla_general, title="Resumen General de Recuperación")
        pdf_gen.add_table(tabla_detalle, title="Detalle de Recuperación")
        # --- Pestaña 2: Distribución de Recuperaciones ---
    with tab2:
        st.subheader("Distribución de Recuperaciones por Día")
        # Boxplot con límites de aceptación
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=datos, x='Día', y='Recuperación (%)', palette='viridis', ax=ax_box)
        ax_box.axhline(98, color='red', linestyle='--', label='Límite Inferior (98%)')
        ax_box.axhline(102, color='green', linestyle='--', label='Límite Superior (102%)')
        ax_box.set_title("Boxplot de Recuperación (%) por Día")
        ax_box.legend()
        st.pyplot(fig_box)
        pdf_gen.capture_figure(fig_box, "Boxplot_Recuperacion")
        plt.close(fig_box)
        
        # Histograma global
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        sns.histplot(data=datos, x='Recuperación (%)', bins=12, kde=True, hue='Día', palette='viridis', ax=ax_hist)
        ax_hist.set_title("Histograma de Recuperación Global")
        st.pyplot(fig_hist)
        pdf_gen.capture_figure(fig_hist, "Histograma_Recuperacion")
        plt.close(fig_hist)
    
    # --- Pestaña 3: Tendencia de la Recuperación ---
    with tab3:
        st.subheader("Evolución de la Recuperación Media por Día")
        fig_line, ax_line = plt.subplots(figsize=(10, 6))
        dias_sorted = resumen['Día']
        medias = resumen['Media (%)']
        # Extraer y convertir los límites inferior y superior del intervalo
        ci_lower_vals = resumen['IC Media (%)'].apply(
            lambda x: float(x.split('%')[0][1:].strip()) if x != "N/A" else np.nan
        )
        ci_upper_vals = resumen['IC Media (%)'].apply(
            lambda x: float(x.split('%')[1].replace(',', '').replace(']', '').strip()) if x != "N/A" else np.nan
        )
        err_lower = medias - ci_lower_vals
        err_upper = ci_upper_vals - medias
        err = [err_lower, err_upper]
        ax_line.errorbar(dias_sorted, medias, yerr=err, fmt='-o', capsize=5, color='blue', label='Media Recuperación')
        ax_line.axhline(98, color='red', linestyle='--', label='Límite Inferior (98%)')
        ax_line.axhline(102, color='green', linestyle='--', label='Límite Superior (102%)')
        ax_line.set_title("Tendencia de la Recuperación Media")
        ax_line.set_xlabel("Día")
        ax_line.set_ylabel("Recuperación (%)")
        ax_line.legend()
        st.pyplot(fig_line)
        pdf_gen.capture_figure(fig_line, "Tendencia_Recuperacion")
        plt.close(fig_line)
    
    # --- Pestaña 4: Datos Detallados ---
    with tab4:
        st.subheader("Datos Completos")
        st.dataframe(
            datos.style.format({
                'Concentración Teórica': '{:.2f}',
                'Concentración Real': '{:.2f}',
                'Recuperación (%)': '{:.2f}%'
            }).apply(lambda row: ['color: #2ecc71' if 98 <= row['Recuperación (%)'] <= 102 else 'color: #e74c3c' for _ in row], axis=1)
        )
        # Agregar una muestra (primeras 10 filas) al PDF con redondeo a 2 decimales
        datos_tabla = datos.head(10).copy()
        tabla_data = [list(datos_tabla.columns)]
        for _, row in datos_tabla.iterrows():
            fila = []
            for valor in row:
                if isinstance(valor, (int, float)):
                    fila.append(f"{valor:<15.2f}")
                else:
                    fila.append(f"{str(valor):<15}")
            tabla_data.append(fila)
        pdf_gen.add_table(tabla_data, title="Datos Detallados (Primeras 10 filas)")
        
        # Botón de descarga (se asume que la función generar_descarga está definida)
        generar_descarga(datos)
    
    return True

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

def evaluar_robustez(datos, pdf_gen):
    """
    Evalúa la robustez del método comparando la media de la respuesta en cada nivel
    de uno o varios factores con la media global. Se calcula la diferencia porcentual 
    absoluta y se determina si dicha variación está dentro de un umbral preestablecido.
    
    Parámetros:
      - datos (DataFrame): Debe contener las columnas ['Respuesta', 'Día', 'Concentración', 'Tipo', 'Analista'].
      - pdf_gen: Objeto para generar el PDF (métodos: add_section_title, add_text_block, add_table, capture_figure).
    
    Metodología (adaptada de la guía CCYAC y Farmacéuticos):
      1. Para cada factor seleccionado, se calcula la media global de 'Respuesta'.
      2. Se agrupa el conjunto de datos por el factor y se calcula la media en cada nivel.
      3. Se calcula la diferencia porcentual absoluta:
            d_i = |(Media del grupo - Media global) / Media global| * 100
      4. Se evalúa robustez si, para cada factor, el valor máximo (o promedio) de d_i es menor o igual al umbral:
            - Por ejemplo, umbral = 2% para métodos cromatográficos/volumétricos,
                          3% para métodos químicos/espectrofotométricos,
                          5% para métodos biológicos.
    
    Los resultados se muestran en Streamlit y se registran en el PDF.
    """
    sns.set_theme(style="whitegrid", palette="pastel")
    st.header("🧪 Análisis de Robustez")
    pdf_gen.add_section_title("Análisis de Robustez")
    
    # Mostrar metodología (adaptada)
    with st.expander("📚 Metodología", expanded=False):
        metodologia = (
            "**Metodología para Evaluar Robustez:**\n"
            "1. Calcular la media global de 'Respuesta' (valor de referencia).\n"
            "2. Para cada nivel del factor a evaluar, calcular la media de 'Respuesta'.\n"
            "3. Calcular la diferencia porcentual absoluta respecto a la media global:\n"
            "      d_i = |(Media del grupo - Media global) / Media global| * 100\n"
            "4. El método se considera robusto respecto a ese factor si la diferencia máxima (o promedio) "
            "es ≤ umbral (ej. 2% para métodos cromatográficos, 3% para métodos químicos, 5% para biológicos).\n"
            "5. Se reportan estos resultados y se generan gráficos de la distribución de 'Respuesta' por nivel."
        )
        st.markdown(metodologia)
        pdf_gen.add_subsection("Metodología")
        pdf_gen.add_text_block(metodologia)
    
    # Validar columnas
    required_cols = ['Respuesta', 'Día', 'Concentración', 'Tipo', 'Analista', 'pH']
    for col in required_cols:
        if col not in datos.columns:
            st.error(f"Falta la columna '{col}' en el DataFrame.")
            pdf_gen.add_text_block(f"✘ Error: Falta la columna '{col}' en el DataFrame.", style="error")
            return False

    # Conversión numérica
    for col in ['Respuesta', 'Concentración']:
        datos[col] = pd.to_numeric(datos[col], errors='coerce')
    datos.dropna(subset=['Respuesta', 'Concentración'], inplace=True)
    
    # Permitir al usuario seleccionar uno o varios factores para evaluar robustez
    opciones = ['Día', 'Concentración', 'Analista','pH']  # Evitamos 'Tipo' pues usualmente se usa para clasificar muestras/estándares
    factores_seleccionados = st.multiselect("Selecciona los factores a evaluar en la robustez:", options=opciones, default=['Día'])
    pdf_gen.add_text_block("Factores a evaluar: " + ", ".join(factores_seleccionados))
    
    # Selección del tipo de método para robustez (para definir el umbral)
    tipo_metodo = st.selectbox("Selecciona el tipo de método:", 
                               options=["cromatografico", "quimico", "espectrofotometrico", "biologico"])
    if tipo_metodo == "cromatografico":
        umbral = 2.0
    elif tipo_metodo in ["quimico", "espectrofotometrico"]:
        umbral = 3.0
    elif tipo_metodo == "biologico":
        umbral = 5.0
    else:
        umbral = 2.0

    pdf_gen.add_text_block(f"Tipo de método seleccionado: {tipo_metodo}. Umbral para robustez: {umbral}%.")
    
    # Calcular la media global de la respuesta en el dataset
    media_global = datos['Respuesta'].mean()
    st.write(f"**Media global de 'Respuesta':** {media_global:.4f}")
    pdf_gen.add_text_block(f"Media global de 'Respuesta': {media_global:.4f}")
    
    # Para cada factor seleccionado, agrupar y calcular diferencias relativas
    resumen_resultados = []
    for factor in factores_seleccionados:
        st.subheader(f"Análisis de Robustez - Factor: {factor}")
        pdf_gen.add_subsection(f"Robustez - Factor: {factor}")
        # Agrupar datos por el factor y calcular la media para cada nivel
        grupos = datos.groupby(factor)['Respuesta'].mean().reset_index()
        grupos['Diferencia (%)'] = abs((grupos['Respuesta'] - media_global) / media_global * 100)
        st.dataframe(grupos)
        
        # Graficar: Boxplot de 'Respuesta' por nivel del factor
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=factor, y='Respuesta', data=datos, ax=ax)
        ax.axhline(media_global, color='red', linestyle='--', label=f'Media Global ({media_global:.2f})')
        ax.set_title(f"Distribución de 'Respuesta' por {factor}")
        ax.legend()
        st.pyplot(fig)
        pdf_gen.capture_figure(fig, f"robustez_boxplot_{factor}")
        plt.close(fig)
        
        # Evaluar robustez: Se considera robusto si la diferencia máxima (o promedio) es ≤ umbral.
        diffs = grupos['Diferencia (%)']
        max_diff = diffs.max()
        avg_diff = diffs.mean()
        robusto = max_diff <= umbral
        st.markdown(
            f"**Resultados para el factor {factor}:**\n"
            f"- Diferencia máxima: {max_diff:.2f}%\n"
            f"- Diferencia promedio: {avg_diff:.2f}%\n"
            f"- **Conclusión:** {'Método Robusto' if robusto else 'Requiere Atención'}"
        )
        pdf_gen.add_text_block(
            f"Resultados para el factor {factor}:\n"
            f"- Diferencia máxima: {max_diff:.2f}%\n"
            f"- Diferencia promedio: {avg_diff:.2f}%\n"
            f"Conclusión: {'Método Robusto' if robusto else 'Requiere Atención'}."
        )
        resumen_resultados.append([factor, f"{max_diff:.2f}%", f"{avg_diff:.2f}%", "Robusto" if robusto else "No Robusto"])
    
    # Agregar un resumen general al PDF
    if resumen_resultados:
        tabla_resumen = [["Factor", "Dif. Máx", "Dif. Promedio", "Robustez"]]
        tabla_resumen.extend(resumen_resultados)
        pdf_gen.add_table(tabla_resumen, title="Resumen de Robustez por Factor")
    
    return True
def agrupar_valores(valores, umbral=1.0):
    """
    Agrupa una lista ordenada de valores en grupos donde la diferencia entre elementos consecutivos es <= umbral.
    Devuelve una lista de grupos (cada grupo es una lista de valores).
    """
    grupos = []
    if not valores:
        return grupos
    grupo_actual = [valores[0]]
    for v in valores[1:]:
        if v - grupo_actual[-1] <= umbral:
            grupo_actual.append(v)
        else:
            grupos.append(grupo_actual)
            grupo_actual = [v]
    grupos.append(grupo_actual)
    return grupos

def evaluar_estabilidad(datos, pdf_gen, test_type="assay"):
    """
    Evalúa la estabilidad del método comparando la respuesta analítica a lo largo del tiempo 
    con la medición inicial (baseline). Se agrupan automáticamente los valores de 'Concentración'
    utilizando una tolerancia de ±1 y, para cada grupo, se calcula el % de recuperación, la diferencia 
    respecto a 100% y el coeficiente de variación (CV).

    Además, se permite seleccionar interactívamente el tipo de método para definir el umbral de aceptación:
       - Cromatográfico: umbral de 2%
       - Químico: umbral de 3%
       - Biológico: umbral de 5%
       
    Parámetros:
      - datos (DataFrame): Debe contener al menos las columnas ['Día', 'Respuesta', 'Concentración'].
         Se asume que 'Día' es numérico o de tipo datetime (en cuyo caso se convierte a días transcurridos).
      - pdf_gen: Objeto para generar el PDF (con métodos como add_section_title, add_text_block, add_table, capture_figure, etc.).
      - test_type (str): Tipo de test; se incluye para compatibilidad (por defecto "assay").

    La función genera gráficos y tablas en Streamlit y documenta los resultados en el PDF.
    """
    sns.set_theme(style="whitegrid", palette="pastel")
    st.header("🧪 Análisis de Estabilidad por Grupo de Concentración")
    pdf_gen.add_section_title("Análisis de Estabilidad por Grupo de Concentración")
    pdf_gen.add_text_block(
        "Este reporte evalúa la estabilidad del método midiendo la respuesta analítica a lo largo del tiempo "
        "y comparándola con la medición inicial (baseline). Los datos se agrupan automáticamente en rangos de "
        "concentración utilizando una tolerancia de ±1 unidad. Para cada rango se calcula el % de recuperación, la "
        "diferencia respecto a 100% y el CV. Se permite seleccionar el tipo de método para definir el umbral de "
        "aceptación."
    )

    # 1. Validar columnas requeridas
    required_cols = ['Día', 'Respuesta', 'Concentración']
    for col in required_cols:
        if col not in datos.columns:
            st.error(f"Falta la columna '{col}' en el DataFrame.")
            pdf_gen.add_text_block(f"✘ Error: Falta la columna '{col}' en el DataFrame.", style="error")
            return False

    # 2. Conversión numérica y limpieza
    for col in ['Respuesta', 'Concentración']:
        datos[col] = pd.to_numeric(datos[col], errors='coerce')
    datos.dropna(subset=['Respuesta', 'Concentración'], inplace=True)

    # 3. Procesar la columna 'Día'
    try:
        if np.issubdtype(datos['Día'].dtype, np.datetime64):
            datos['Día_num'] = (datos['Día'] - datos['Día'].min()).dt.days
        else:
            datos['Día_num'] = pd.to_numeric(datos['Día'], errors='coerce')
    except Exception as e:
        st.error(f"Error al convertir 'Día': {str(e)}")
        pdf_gen.add_text_block(f"✘ Error al convertir 'Día': {str(e)}", style="error")
        return False
    datos.dropna(subset=['Día_num'], inplace=True)

    # 4. Agrupar automáticamente los valores de concentración usando tolerancia ±1
    unique_vals = sorted(datos['Concentración'].unique())
    grupos_conc = agrupar_valores(unique_vals, umbral=1.0)
    mapping = {}
    for grupo in grupos_conc:
        rep = np.median(grupo)
        for v in grupo:
            mapping[v] = rep
    datos['Rango'] = datos['Concentración'].map(mapping).apply(lambda x: f"{x:.2f}")

    # Mostrar la agrupación para depuración
    st.subheader("Agrupación de Concentraciones")
    df_grupos = datos[['Concentración', 'Rango']].drop_duplicates().sort_values(by='Concentración')
    st.dataframe(df_grupos)
    tabla_grupos = [["Concentración", "Rango asignado"]]
    for _, row in df_grupos.iterrows():
        tabla_grupos.append([f"{row['Concentración']:.2f}", row['Rango']])
    pdf_gen.add_table(tabla_grupos, title="Agrupación de Concentraciones (Automática)")

    # 5. Seleccionar el tipo de método (para establecer umbral)
    metodo_seleccionado = st.selectbox(
        "Seleccione el tipo de método:",
        options=["cromatografico", "quimico", "biologico"],
        index=0
    )
    if metodo_seleccionado == "cromatografico":
        umbral = 2.0
    elif metodo_seleccionado == "quimico":
        umbral = 3.0
    elif metodo_seleccionado == "biologico":
        umbral = 5.0
    else:
        umbral = 2.0
    st.write(f"Umbral de aceptación para el método {metodo_seleccionado}: {umbral}%")
    pdf_gen.add_text_block(f"Tipo de método seleccionado: {metodo_seleccionado}. Umbral de aceptación: {umbral}%.", style="info")

    # 6. Permitir al usuario seleccionar un factor variable adicional (opcional)
    columnas_factor = [col for col in datos.columns if col not in ['Día', 'Día_num', 'Respuesta', 'Concentración', 'Rango']]
    if columnas_factor:
        factor_variable = st.selectbox("Seleccione un factor variable adicional (opcional):", ["Ninguno"] + columnas_factor)
    else:
        factor_variable = "Ninguno"
    pdf_gen.add_text_block(f"Factor variable seleccionado: {factor_variable}", style="info")

    # 7. Iterar sobre cada rango y realizar el análisis de estabilidad
    resultados_totales = []
    rangos_disponibles = sorted(datos['Rango'].unique(), key=lambda x: float(x))
    for rango_label in rangos_disponibles:
        df_rango = datos[datos['Rango'] == rango_label].copy()
        if df_rango.empty:
            continue

        st.markdown(f"### Análisis para el Rango {rango_label}")
        pdf_gen.add_subsection(f"Rango {rango_label}")

        # Agrupar por día en este rango
        resumen = df_rango.groupby('Día_num')['Respuesta'].agg(['mean', 'std', 'count']).reset_index()
        resumen.rename(columns={'mean': 'Media', 'std': 'DE', 'count': 'N'}, inplace=True)

        # Baseline: se toma la medición del día inicial
        baseline_row = resumen.loc[resumen['Día_num'].idxmin()]
        baseline = baseline_row['Media']
        if baseline == 0:
            st.error("La media en el día inicial es 0, no es posible calcular la recuperación.")
            pdf_gen.add_text_block("✘ Error: La media en el tiempo inicial es 0.", style="error")
            continue

        resumen['Recuperación (%)'] = (resumen['Media'] / baseline) * 100
        resumen['Diferencia (%)'] = abs(resumen['Recuperación (%)'] - 100)
        resumen['CV'] = (resumen['DE'] / resumen['Media']) * 100

        st.dataframe(resumen)
        
        # Gráficos para este rango
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        sns.lineplot(data=resumen, x='Día_num', y='Recuperación (%)', marker='o', ax=ax[0])
        ax[0].axhline(100, color='red', linestyle='--', label="100% (Baseline)")
        ax[0].axhline(100+umbral, color='gray', linestyle='--', label=f"100% ± {umbral}%")
        ax[0].axhline(100-umbral, color='gray', linestyle='--')
        ax[0].set_title(f"Recuperación (%) vs. Tiempo - Rango {rango_label}")
        ax[0].set_xlabel("Día (numérico)")
        ax[0].set_ylabel("Recuperación (%)")
        ax[0].legend()
        
        sns.barplot(data=resumen, x='Día_num', y='CV', ax=ax[1], palette='Blues', edgecolor='black')
        ax[1].axhline(umbral, color='red', linestyle='--', label=f"Umbral {umbral}%")
        ax[1].set_title(f"Variabilidad (CV) - Rango {rango_label}")
        ax[1].set_xlabel("Día (numérico)")
        ax[1].set_ylabel("CV (%)")
        ax[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        pdf_gen.capture_figure(fig, f"estabilidad_{rango_label}")
        plt.close(fig)
        
        # Análisis complementario del factor variable, si se seleccionó
        if factor_variable != "Ninguno":
            st.markdown(f"#### Análisis del factor variable '{factor_variable}' en el rango {rango_label}")
            pdf_gen.add_text_block(f"Análisis del factor variable '{factor_variable}' en el rango {rango_label}.", style="info")
            df_factor = df_rango.dropna(subset=[factor_variable])
            resumen_factor = df_factor.groupby(factor_variable)['Respuesta'].mean().reset_index()
            st.dataframe(resumen_factor)
            fig_factor, ax_factor = plt.subplots(figsize=(6, 4))
            sns.barplot(data=resumen_factor, x=factor_variable, y='Respuesta', ax=ax_factor, palette='viridis')
            ax_factor.set_title(f"Media de Respuesta por '{factor_variable}' - Rango {rango_label}")
            plt.xticks(rotation=45)
            st.pyplot(fig_factor)
            pdf_gen.capture_figure(fig_factor, f"factor_{factor_variable}_{rango_label}")
            plt.close(fig_factor)
        
        # Registro de métricas clave para este rango
        resultado = {
            'Rango': rango_label,
            'Recuperación Mínima (%)': f"{resumen['Recuperación (%)'].min():.2f}%",
            'Diferencia Máxima (%)': f"{resumen['Diferencia (%)'].max():.2f}%",
            'CV Máximo (%)': f"{resumen['CV'].max():.2f}%"
        }
        resultados_totales.append(resultado)
    
    # Reporte consolidado
    if resultados_totales:
        tabla_header = ["Rango", "Recup. Mínima", "Dif. Máxima", "CV Máximo"]
        tabla_data = [[r['Rango'], r['Recuperación Mínima (%)'], r['Diferencia Máxima (%)'], r['CV Máximo (%)']] for r in resultados_totales]
        pdf_gen.add_table([tabla_header] + tabla_data, title="Resumen de Estabilidad por Rango")
        todas_estables = all(float(r['Diferencia Máxima (%)'].replace('%','')) <= umbral for r in resultados_totales)
        conclusion = "El método es estable en todos los rangos." if todas_estables else "El método presenta variaciones superiores al umbral en algunos rangos."
        st.markdown(f"### Conclusión: {conclusion}")
        pdf_gen.add_text_block(f"Conclusión de estabilidad: {conclusion}", style="info")
        return resultados_totales
    else:
        st.warning("No se pudieron obtener resultados de estabilidad en ningún rango.")
        pdf_gen.add_text_block("⚠️ No se obtuvieron resultados de estabilidad en ningún rango.", style="warning")
        return False
    
if modulo == "Linealidad y Rango":
    st.header("Análisis de Linealidad y Rango")
    st.info("""
        **Datos requeridos para este módulo:**
        - **Concentración:** Concentraciones de las soluciones estándar.
        - **Respuesta:** Valores de Respuesta medidos.
        - **Tipo:** Identificar si es "Estándar" o "Muestra".
        - **Día:** Identificar el día de la medición.""")  

    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Respuesta', 'Tipo'")

    
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    if procesar_archivo(datos, calcular_linealidad, modulo):
        st.success("Análisis completado!")

    pdf = st.session_state.get('current_pdf')
    if pdf:
            st.download_button(
                label="⬇️ Descargar Reporte Completo",
                data=pdf,
                file_name=f"Reporte_{modulo.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
    

elif modulo == "Límites de Detección y Cuantificación":
    st.header("Cálculo de LOD y LOQ")
    st.info("""
        **Datos requeridos para este módulo:**
        - **Concentración:** Concentraciones de las soluciones estándar.
        - **Respuesta:** Valores de Respuesta medidos.
        - **Tipo:** Identificar si es "Estándar" o "Muestra".
        - **Día:** Día en que se realizó la medición.
        """)  

    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Respuesta', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    
    if datos:
        if procesar_archivo(datos, [calcular_lod_loq, graficar_curva_calibracion_streamlit], "Límites de Detección y Cuantificación"):
            st.success("¡Análisis completado!")
            if 'current_pdf' in st.session_state:
                st.download_button(
                    label="Descargar PDF",
                    data=st.session_state.current_pdf,
                    file_name="reporte_limites.pdf",
                    mime="application/pdf"
                )


# Módulo de Precisión

elif modulo == "Precisión (Repetibilidad e Intermedia)":
    st.header("Evaluación de Precisión")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Respuesta:** Datos de Respuesta agrupados por días y repeticiones.
        """
    )
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Respuesta', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    
    if datos:
        if procesar_archivo(datos, [calcular_precision_por_rango], "Precisión (Repetibilidad e Intermedia)"):
            st.success("¡Análisis completado!")
            if 'current_pdf' in st.session_state:
                st.download_button(
                    label="Descargar PDF",
                    data=st.session_state.current_pdf,
                    file_name="reporte_Precisión.pdf",
                    mime="application/pdf"
                )
    

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
    
    if datos:
        if procesar_archivo(datos, [calcular_exactitud], "Exactitud y recuperación"):
            st.success("¡Análisis completado!")
            if 'current_pdf' in st.session_state:
                st.download_button(
                    label="Descargar PDF",
                    data=st.session_state.current_pdf,
                    file_name="reporte_exactitud.pdf",
                    mime="application/pdf"
                )

# Módulo de Robustez
elif modulo == "Robustez":
    st.header("Evaluación de Robustez")
    st.info("""
        **Datos requeridos para este módulo:**
        - **Factores variables:** Datos que representan condiciones variables del experimento.
        - **Resultados:** Datos de resultados obtenidos bajo dichas condiciones.
        """) 
    img_path = imagenes_dir / "muestra.png"
    st.image(str(img_path), caption="Estructura requerida: Columnas 'Día', 'Concentración', 'Respuesta', 'Tipo'")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    
    if datos:
        if procesar_archivo(datos, [evaluar_robustez], "Robustez del Metodo"):
            st.success("¡Análisis completado!")
            if 'current_pdf' in st.session_state:
                st.download_button(
                    label="Descargar PDF",
                    data=st.session_state.current_pdf,
                    file_name="reporte_robustez.pdf",
                    mime="application/pdf"
                )

elif modulo == "Estabilidad":
    st.header("Evaluación de Estabilidad")
    st.info("""
        **Datos requeridos para este módulo:**
        - **Día:** Día de la medición (numérico o fecha)
        - **Respuesta:** Valores de respuesta analítica
        - **Tipo:** Clasificación de la muestra (Ej: Estándar, Muestra)
        - **Factores Variables:** Columnas adicionales para análisis multivariable
        """)

    st.image(str(imagenes_dir / "muestra.png"), caption="Estructura requerida")
    datos = st.file_uploader("Sube tu archivo:", type=['csv', 'xlsx'])
    
    if datos:
        # Procesar con manejo de PDF integrado
        if procesar_archivo(datos, [evaluar_estabilidad], "Estabilidad del Método"):
            st.success("¡Análisis completado!")
            if 'current_pdf' in st.session_state:
                st.download_button(
                    label="Descargar PDF",
                    data=st.session_state.current_pdf.getvalue(),
                    file_name="reporte_estabilidad.pdf",
                    mime="application/pdf"
                )