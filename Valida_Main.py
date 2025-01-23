import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, f_oneway
import streamlit as st

# Configuración general de la aplicación
st.set_page_config(page_title="Validación de Métodos Analíticos - Espectrofotometría UV-Vis", layout="wide")
st.title("Validación de Métodos Analíticos para Espectrofotometría UV-Vis (ICHQ2)")

# Módulos disponibles
modulo = st.sidebar.selectbox(
    "Selecciona el módulo de análisis:",
    [
        "Linealidad y Rango",
        "Límites de Detección y Cuantificación",
        "Exactitud (Recuperación)",
        "Precisión (Repetibilidad e Intermedia)",
        "Robustez",
        "Especificidad",
        "Estabilidad de la Solución"
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

# Funciones específicas por módulo

def calcular_linealidad(datos):
    """Calcula la linealidad y rango del método."""
    columnas_necesarias = ['Concentración', 'Absorbancia', 'Tipo']
    if not validar_columnas(datos, columnas_necesarias):
        return

    # Agrupación por tipo (Estándar o Muestra)
    grouped = datos.groupby('Tipo')

    # Datos del Estándar
    estandar = grouped.get_group('Estándar')
    estandar_promedio = estandar.groupby('Concentración')['Absorbancia'].mean().reset_index()
    x_estandar = estandar_promedio['Concentración']
    y_estandar = estandar_promedio['Absorbancia']

    # Cálculo de la regresión lineal
    slope, intercept, r_value, p_value, std_err = linregress(x_estandar, y_estandar)
    st.write(f"**Pendiente (Slope):** {slope:.4f}")
    st.write(f"**Intercepto (Intercept):** {intercept:.4f}")
    st.write(f"**Coeficiente de determinación (R²):** {r_value**2:.4f}")
    st.write(f"**Valor p:** {p_value:.4e}")

    # Validación de la linealidad
    if r_value**2 >= 0.995:
        st.success("Cumple con los criterios de linealidad (R² ≥ 0.995).")
    else:
        st.error("No cumple con los criterios de linealidad (R² < 0.995).")

    # Gráfica para Estándar
    plt.figure(figsize=(8, 5))
    sns.regplot(x=x_estandar, y=y_estandar, ci=None, line_kws={'color': 'red'})
    plt.title("Linealidad: Concentración vs Absorbancia (Estándar Promedio)")
    plt.xlabel("Concentración")
    plt.ylabel("Absorbancia")
    st.pyplot(plt)

    # Datos de la Muestra
    muestra = grouped.get_group('Muestra')
    x_muestra = muestra['Absorbancia']
    concentraciones_estimadas = (x_muestra - intercept) / slope
    muestra['Concentración Estimada'] = concentraciones_estimadas

    # Gráfica para Muestra
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=muestra['Concentración Estimada'], y=muestra['Absorbancia'], color='blue', label='Muestra')
    plt.title("Concentración Estimada de la Muestra")
    plt.xlabel("Concentración Estimada")
    plt.ylabel("Absorbancia")
    st.pyplot(plt)

    return muestra

def calcular_lod_loq(datos):
    """Calcula los límites de detección y cuantificación."""
    columnas_necesarias = ['Ruido', 'Pendiente']
    if not validar_columnas(datos, columnas_necesarias):
        return

    std_dev = datos['Ruido'].std()
    slope = datos['Pendiente'].iloc[0]

    lod = (3.3 * std_dev) / slope
    loq = (10 * std_dev) / slope

    st.write(f"**Límite de Detección (LOD):** {lod:.4f}")
    st.write(f"**Límite de Cuantificación (LOQ):** {loq:.4f}")

def calcular_precision(datos):
    """Evalúa la precisión mediante el cálculo del RSD."""
    columnas_necesarias = ['Día', 'Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return

    grupos = datos.groupby('Día')
    rsd_por_grupo = grupos['Absorbancia'].std() / grupos['Absorbancia'].mean() * 100
    rsd_general = datos['Absorbancia'].std() / datos['Absorbancia'].mean() * 100

    st.write("**RSD por día:**")
    st.table(rsd_por_grupo)
    st.write(f"**RSD General:** {rsd_general:.2f}%")

    if rsd_general <= 2:
        st.success("Cumple con los criterios de precisión (RSD ≤ 2%).")
    else:
        st.error("No cumple con los criterios de precisión (RSD > 2%).")

def evaluar_robustez(datos):
    """Evalúa la robustez del método analítico mediante ANOVA."""
    columnas_necesarias = ['Absorbancia']
    if not validar_columnas(datos, columnas_necesarias):
        return

    factor = st.selectbox("Selecciona el factor a evaluar:", datos.columns[1:])
    if factor not in datos.columns:
        st.error("El factor seleccionado no está en los datos.")
        return

    grupos = [grupo['Absorbancia'].values for _, grupo in datos.groupby(factor)]
    estadistico, p_value = f_oneway(*grupos)

    st.write(f"**Valor p:** {p_value:.4e}")

    if p_value > 0.05:
        st.success("No hay diferencias significativas (p > 0.05). El método es robusto.")
    else:
        st.error("Hay diferencias significativas (p ≤ 0.05). El método no es robusto.")

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
if modulo == "Análisis de Linealidad y Rango":
    st.header("Análisis de Linealidad y Rango")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Concentración:** Concentraciones de las soluciones estándar.
        - **Absorbancia:** Valores de absorbancia medidos.
        - **Tipo:** Identificar si es "Estándar" o "Muestra".
        """
    )
    datos = st.file_uploader("Sube un archivo con datos de Concentración y Absorbancia:", type=['csv', 'xlsx'])
    if datos:
        try:
            datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
            previsualizar_datos(datos_df)
            calcular_linealidad(datos_df)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

elif modulo == "Límites de Detección y Cuantificación":
    st.header("Cálculo de LOD y LOQ")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Ruido:** Datos del ruido del sistema.
        - **Pendiente:** Datos de la pendiente de la curva de calibración.
        """
    )
    datos = st.file_uploader("Sube un archivo con datos de Ruido y Pendiente:", type=['csv', 'xlsx'])
    if datos:
        try:
            datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
            previsualizar_datos(datos_df)
            calcular_lod_loq(datos_df)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

elif modulo == "Precisión (Repetibilidad e Intermedia)":
    st.header("Evaluación de Precisión")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Absorbancia:** Datos de absorbancia agrupados por días y repeticiones.
        """
    )
    datos = st.file_uploader("Sube un archivo con datos de Absorbancia agrupados por Día:", type=['csv', 'xlsx'])
    if datos:
        try:
            datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
            previsualizar_datos(datos_df)
            calcular_precision(datos_df)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

elif modulo == "Robustez":
    st.header("Evaluación de Robustez")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Factores variables:** Datos que representan condiciones variables del experimento.
        - **Resultados:** Datos de resultados obtenidos bajo dichas condiciones.
        """
    )
    datos = st.file_uploader("Sube un archivo con datos para evaluar robustez:", type=['csv', 'xlsx'])
    if datos:
        try:
            datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
            previsualizar_datos(datos_df)
            evaluar_robustez(datos_df)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

elif modulo == "Especificidad":
    st.header("Evaluación de Especificidad")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Selectividad:** Datos que evidencien la capacidad para analizar específicamente el analito en presencia de interferencias.
        """
    )
    datos = st.file_uploader("Sube un archivo con datos para evaluar especificidad:", type=['csv', 'xlsx'])
    if datos:
        try:
            datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
            previsualizar_datos(datos_df)
            st.warning("La evaluación de especificidad aún no está implementada.")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

elif modulo == "Estabilidad de la Solución":
    st.header("Evaluación de Estabilidad de la Solución")
    st.info(
        """
        **Datos requeridos para este módulo:**
        - **Tiempo:** Tiempos de medición.
        - **Absorbancia:** Datos de absorbancia medidos en cada intervalo de tiempo.
        """
    )
    datos = st.file_uploader("Sube un archivo con datos de Tiempo y Absorbancia:", type=['csv', 'xlsx'])
    if datos:
        try:
            datos_df = pd.read_csv(datos) if datos.name.endswith('.csv') else pd.read_excel(datos)
            previsualizar_datos(datos_df)
            evaluar_estabilidad(datos_df)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
